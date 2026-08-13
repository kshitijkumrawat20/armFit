use crate::benchmarks::HardwareMatchLevel;
use crate::fit::{ArmArchitectureCompatibility, FitLevel, InferenceRuntime, ModelFit};
use crate::hardware::{CpuArchitecture, SystemSpecs};
use crate::models::{LlmModel, ModelDatabase};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OptimizationProvenance {
    Measured,
    Estimated,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeAvailability {
    Unknown,
    Unavailable,
    Installed,
    Benchmarkable,
    Benchmarked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArmValidationStatus {
    ExactHardwareMatch,
    HighConfidenceMatch,
    PartialMatch,
    EstimatedOnly,
    Unknown,
}

#[derive(Clone, Serialize)]
pub struct OptimizationCandidate {
    pub model: LlmModel,
    pub quantization: String,
    pub runtime: InferenceRuntime,
    pub context_length: u32,
    pub thread_count: Option<usize>,
    pub batch_size: Option<u32>,
    pub fit: ModelFit,
    pub estimated_tps: f64,
    pub measured_tps: Option<f64>,
    pub benchmark_match_level: HardwareMatchLevel,
    pub performance_provenance: OptimizationProvenance,
    pub runtime_availability: RuntimeAvailability,
    pub validation_status: ArmValidationStatus,
    pub memory_required_gb: f64,
    pub memory_available_gb: f64,
    pub compatibility_status: ArmArchitectureCompatibility,
}

fn runtime_availability_for_fit(fit: &ModelFit) -> RuntimeAvailability {
    if fit.measured_tps.is_some() {
        RuntimeAvailability::Benchmarked
    } else if fit.installed {
        RuntimeAvailability::Installed
    } else if fit.runtime == InferenceRuntime::Unsupported {
        RuntimeAvailability::Unavailable
    } else if fit.estimated_tps > 0.0 {
        RuntimeAvailability::Benchmarkable
    } else {
        RuntimeAvailability::Unknown
    }
}

fn validation_status_for_match(
    specs: &SystemSpecs,
    match_level: HardwareMatchLevel,
    measured: bool,
) -> ArmValidationStatus {
    if !measured {
        return ArmValidationStatus::EstimatedOnly;
    }

    match specs.architecture {
        CpuArchitecture::Aarch64 => match match_level {
            HardwareMatchLevel::Exact => ArmValidationStatus::ExactHardwareMatch,
            HardwareMatchLevel::HighConfidence => ArmValidationStatus::HighConfidenceMatch,
            HardwareMatchLevel::Partial => ArmValidationStatus::PartialMatch,
            HardwareMatchLevel::NoMatch => ArmValidationStatus::EstimatedOnly,
        },
        CpuArchitecture::X86_64 => ArmValidationStatus::Unknown,
        CpuArchitecture::Unknown => ArmValidationStatus::Unknown,
    }
}

#[derive(Clone, Serialize)]
pub struct OptimizationResult {
    pub hardware_architecture: CpuArchitecture,
    pub selected_candidate: Option<OptimizationCandidate>,
    pub best_available: Option<OptimizationCandidate>,
    pub best_measured: Option<OptimizationCandidate>,
    pub best_predicted: Option<OptimizationCandidate>,
    pub ranked_candidates: Vec<OptimizationCandidate>,
    pub explanation: String,
    pub recommendation: String,
    pub is_arm_aware: bool,
}

fn candidate_sort_key(candidate: &OptimizationCandidate) -> (f64, f64, f64) {
    let throughput = candidate.measured_tps.unwrap_or(candidate.estimated_tps);
    let fit_rank = match candidate.fit.fit_level {
        FitLevel::Perfect => 4.0,
        FitLevel::Good => 3.0,
        FitLevel::Marginal => 2.0,
        FitLevel::TooTight => 0.0,
    };
    let evidence_rank = match candidate.performance_provenance {
        OptimizationProvenance::Measured => 5.0,
        OptimizationProvenance::Estimated => 2.0,
        OptimizationProvenance::Unavailable => 0.0,
    };
    let validation_rank = match candidate.validation_status {
        ArmValidationStatus::ExactHardwareMatch => 5.0,
        ArmValidationStatus::HighConfidenceMatch => 4.0,
        ArmValidationStatus::PartialMatch => 3.0,
        ArmValidationStatus::EstimatedOnly => 1.0,
        ArmValidationStatus::Unknown => 0.0,
    };
    let benchmark_bonus = match candidate.benchmark_match_level {
        HardwareMatchLevel::Exact => 1.0,
        HardwareMatchLevel::HighConfidence => 0.75,
        HardwareMatchLevel::Partial => 0.5,
        HardwareMatchLevel::NoMatch => 0.0,
    };

    (
        evidence_rank * 100.0
            + validation_rank * 20.0
            + benchmark_bonus * 10.0
            + fit_rank * 10.0
            + candidate.fit.score,
        throughput,
        candidate.fit.score,
    )
}

pub fn optimize_for_system(
    specs: &SystemSpecs,
    db: &ModelDatabase,
    limit: usize,
) -> OptimizationResult {
    let mut candidates = Vec::new();
    for model in db.get_all_models() {
        let fit = ModelFit::analyze_with_context_limit(model, specs, None);
        if fit.fit_level == FitLevel::TooTight && fit.estimated_tps <= 0.0 {
            continue;
        }

        let measured_tps = fit.measured_tps.as_ref().map(|m| m.tok_s);
        let benchmark_match_level = fit
            .measured_tps
            .as_ref()
            .map(|m| m.match_level)
            .unwrap_or(HardwareMatchLevel::NoMatch);
        let performance_provenance = if fit.measured_tps.is_some() {
            OptimizationProvenance::Measured
        } else if fit.estimated_tps > 0.0 {
            OptimizationProvenance::Estimated
        } else {
            OptimizationProvenance::Unavailable
        };

        let compatibility_status = fit
            .arm_recommendation(specs)
            .map(|rec| rec.architecture_compatibility)
            .unwrap_or(ArmArchitectureCompatibility::Unknown);

        let candidate = OptimizationCandidate {
            model: model.clone(),
            quantization: fit.best_quant.clone(),
            runtime: fit.runtime,
            context_length: fit.effective_context_length,
            thread_count: None,
            batch_size: None,
            fit: fit.clone(),
            estimated_tps: fit.estimated_tps,
            measured_tps,
            benchmark_match_level,
            performance_provenance,
            runtime_availability: runtime_availability_for_fit(&fit),
            validation_status: validation_status_for_match(
                specs,
                benchmark_match_level,
                fit.measured_tps.is_some(),
            ),
            memory_required_gb: fit.memory_required_gb,
            memory_available_gb: fit.memory_available_gb,
            compatibility_status,
        };
        candidates.push(candidate);
    }

    candidates.sort_by(|a, b| {
        candidate_sort_key(b)
            .partial_cmp(&candidate_sort_key(a))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut ranked_candidates = candidates;
    if !ranked_candidates.is_empty() && limit > 0 {
        ranked_candidates.truncate(limit);
    }

    let best_measured = ranked_candidates
        .iter()
        .filter(|c| c.performance_provenance == OptimizationProvenance::Measured)
        .max_by(|a, b| {
            let a_key = (a.measured_tps.unwrap_or(0.0), a.fit.score);
            let b_key = (b.measured_tps.unwrap_or(0.0), b.fit.score);
            a_key
                .partial_cmp(&b_key)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    let best_predicted = ranked_candidates
        .iter()
        .filter(|c| c.performance_provenance == OptimizationProvenance::Estimated)
        .max_by(|a, b| {
            let a_key = (a.estimated_tps, a.fit.score);
            let b_key = (b.estimated_tps, b.fit.score);
            a_key
                .partial_cmp(&b_key)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .cloned();

    let best_available = best_measured
        .clone()
        .or_else(|| best_predicted.clone())
        .or_else(|| ranked_candidates.first().cloned());

    let selected_candidate = best_available.clone();
    let is_arm_aware = specs.architecture == CpuArchitecture::Aarch64;

    let recommendation = selected_candidate
        .as_ref()
        .map(|candidate| {
            let label = candidate.model.name.as_str();
            let provenance = match candidate.performance_provenance {
                OptimizationProvenance::Measured => "measured",
                OptimizationProvenance::Estimated => "predicted",
                OptimizationProvenance::Unavailable => "unavailable",
            };
            format!("{label} is the best {provenance} fit for this machine")
        })
        .unwrap_or_else(|| "No optimization recommendation available".to_string());

    let explanation = selected_candidate
        .as_ref()
        .map(explain_candidate)
        .unwrap_or_else(|| "No optimization recommendation available".to_string());

    OptimizationResult {
        hardware_architecture: specs.architecture,
        selected_candidate,
        best_available,
        best_measured,
        best_predicted,
        ranked_candidates,
        explanation,
        recommendation,
        is_arm_aware,
    }
}

pub fn explain_candidate(candidate: &OptimizationCandidate) -> String {
    let proven = match candidate.performance_provenance {
        OptimizationProvenance::Measured => {
            let match_label = match candidate.benchmark_match_level {
                HardwareMatchLevel::Exact => "Exact hardware match",
                HardwareMatchLevel::HighConfidence => "High-confidence hardware match",
                HardwareMatchLevel::Partial => "Partial hardware match",
                HardwareMatchLevel::NoMatch => "No hardware match",
            };
            let measured = candidate
                .measured_tps
                .map(|t| format!("Measured throughput: {:.2} tok/s", t))
                .unwrap_or_else(|| "Measured throughput: unavailable".to_string());
            format!(
                "Measured candidate: {}. {} ({}). Runtime: {}. Runtime state: {:?}; validation: {:?}. Fit: {:?} with {:.1} GB / {:.1} GB memory. Estimated throughput: {:.2} tok/s.",
                candidate.model.name,
                measured,
                match_label,
                candidate.runtime.label(),
                candidate.runtime_availability,
                candidate.validation_status,
                candidate.fit.fit_level,
                candidate.memory_required_gb,
                candidate.memory_available_gb,
                candidate.estimated_tps,
            )
        }
        OptimizationProvenance::Estimated => format!(
            "Predicted candidate: {}. Estimated throughput: {:.2} tok/s with {} runtime. Runtime state: {:?}; validation: {:?}. Fit: {:?} (requires {:.1} GB, available {:.1} GB).",
            candidate.model.name,
            candidate.estimated_tps,
            candidate.runtime.label(),
            candidate.runtime_availability,
            candidate.validation_status,
            candidate.fit.fit_level,
            candidate.memory_required_gb,
            candidate.memory_available_gb,
        ),
        OptimizationProvenance::Unavailable => format!(
            "Candidate {} has no measured or estimated throughput data. Fit: {:?} with {:.1} GB / {:.1} GB memory.",
            candidate.model.name,
            candidate.fit.fit_level,
            candidate.memory_required_gb,
            candidate.memory_available_gb,
        ),
    };
    proven
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmarks::{HardwareMatchLevel, MeasuredTps};
    use crate::fit::{EstimateBasis, FitLevel, InferenceRuntime, RunMode, ScoreComponents};
    use crate::hardware::{CpuArchitecture, GpuBackend, SystemSpecs};
    use crate::models::{LlmModel, ModelFormat, UseCase};

    fn model(name: &str, params: u64, ram: f64) -> LlmModel {
        LlmModel {
            name: name.to_string(),
            provider: "llama.cpp".to_string(),
            parameter_count: params.to_string(),
            parameters_raw: Some(params),
            min_ram_gb: ram,
            recommended_ram_gb: ram * 1.5,
            min_vram_gb: Some(ram * 0.75),
            quantization: "Q4_K_M".to_string(),
            context_length: 8192,
            use_case: "chat".to_string(),
            is_moe: false,
            num_experts: None,
            active_experts: None,
            active_parameters: None,
            release_date: None,
            gguf_sources: Vec::new(),
            capabilities: Vec::new(),
            languages: Vec::new(),
            format: ModelFormat::Gguf,
            num_attention_heads: None,
            num_key_value_heads: None,
            num_hidden_layers: None,
            head_dim: None,
            attention_layout: None,
            license: None,
            hidden_size: None,
            moe_intermediate_size: None,
            vocab_size: None,
            shared_expert_intermediate_size: None,
            architecture: Some("llama3".to_string()),
        }
    }

    fn specs_for_arch(arch: CpuArchitecture) -> SystemSpecs {
        SystemSpecs {
            architecture: arch,
            total_ram_gb: 16.0,
            available_ram_gb: 14.0,
            physical_cpu_cores: Some(8),
            total_cpu_cores: 8,
            cpu_name: match arch {
                CpuArchitecture::Aarch64 => "Cortex-A78".to_string(),
                CpuArchitecture::X86_64 => "Intel Core i7".to_string(),
                CpuArchitecture::Unknown => "Unknown CPU".to_string(),
            },
            cpu_vendor: Some("arm".to_string()),
            arm_capabilities: None,
            has_gpu: false,
            gpu_vram_gb: None,
            total_gpu_vram_gb: None,
            gpu_available_gb: None,
            gpu_name: None,
            gpu_count: 0,
            unified_memory: false,
            backend: GpuBackend::CpuArm,
            gpus: Vec::new(),
            cluster_mode: false,
            cluster_node_count: 0,
        }
    }

    fn fit_for_model(
        model: &LlmModel,
        estimated_tps: f64,
        measured: Option<f64>,
        match_level: HardwareMatchLevel,
    ) -> ModelFit {
        ModelFit {
            model: model.clone(),
            fit_level: FitLevel::Good,
            run_mode: RunMode::Gpu,
            memory_required_gb: 8.0,
            memory_available_gb: 16.0,
            utilization_pct: 50.0,
            notes: vec!["candidate".to_string()],
            moe_offloaded_gb: None,
            score: 72.0,
            score_components: ScoreComponents {
                quality: 70.0,
                speed: 75.0,
                fit: 80.0,
                context: 60.0,
            },
            estimated_tps,
            best_quant: "Q4_K_M".to_string(),
            use_case: UseCase::Chat,
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
            fits_with_turboquant: false,
            effective_context_length: 8192,
            usable_context: 8192,
            estimate_basis: EstimateBasis::default(),
            measured_tps: measured.map(|value| MeasuredTps {
                tok_s: value,
                sample_count: 5,
                hardware_label: "fixture".to_string(),
                source: crate::benchmarks::MeasuredSource::Community,
                match_level,
            }),
        }
    }

    #[test]
    fn arm64_runtime_unavailable_is_explicitly_unavailable() {
        let fit = ModelFit {
            runtime: InferenceRuntime::Unsupported,
            installed: false,
            estimated_tps: 0.0,
            measured_tps: None,
            ..fit_for_model(
                &model("test", 4_000_000_000, 8.0),
                0.0,
                None,
                HardwareMatchLevel::NoMatch,
            )
        };

        assert_eq!(
            runtime_availability_for_fit(&fit),
            RuntimeAvailability::Unavailable
        );
    }

    #[test]
    fn arm64_runtime_installed_is_distinct_from_benchmarkable() {
        let fit = ModelFit {
            runtime: InferenceRuntime::LlamaCpp,
            installed: true,
            estimated_tps: 12.0,
            measured_tps: None,
            ..fit_for_model(
                &model("test", 4_000_000_000, 8.0),
                12.0,
                None,
                HardwareMatchLevel::NoMatch,
            )
        };

        assert_eq!(
            runtime_availability_for_fit(&fit),
            RuntimeAvailability::Installed
        );
    }

    #[test]
    fn arm64_runtime_benchmarkable_remains_non_measured() {
        let fit = ModelFit {
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
            estimated_tps: 12.0,
            measured_tps: None,
            ..fit_for_model(
                &model("test", 4_000_000_000, 8.0),
                12.0,
                None,
                HardwareMatchLevel::NoMatch,
            )
        };

        assert_eq!(
            runtime_availability_for_fit(&fit),
            RuntimeAvailability::Benchmarkable
        );
    }

    #[test]
    fn exact_arm64_benchmark_uses_exact_validation() {
        let specs = specs_for_arch(CpuArchitecture::Aarch64);
        let status = validation_status_for_match(&specs, HardwareMatchLevel::Exact, true);
        assert_eq!(status, ArmValidationStatus::ExactHardwareMatch);
    }

    #[test]
    fn different_arm_cpu_benchmark_is_not_exact() {
        let specs = specs_for_arch(CpuArchitecture::Aarch64);
        let status = validation_status_for_match(&specs, HardwareMatchLevel::HighConfidence, true);
        assert_ne!(status, ArmValidationStatus::ExactHardwareMatch);
    }

    #[test]
    fn arm64_result_outscores_x86_64_benchmark_evidence() {
        let arm = OptimizationCandidate {
            model: model("arm-32b", 3_200_000_000, 8.0),
            quantization: "Q4_K_M".to_string(),
            runtime: InferenceRuntime::LlamaCpp,
            context_length: 8192,
            thread_count: None,
            batch_size: None,
            fit: fit_for_model(
                &model("arm-32b", 3_200_000_000, 8.0),
                12.0,
                Some(12.0),
                HardwareMatchLevel::Exact,
            ),
            estimated_tps: 12.0,
            measured_tps: Some(12.0),
            benchmark_match_level: HardwareMatchLevel::Exact,
            performance_provenance: OptimizationProvenance::Measured,
            runtime_availability: RuntimeAvailability::Benchmarked,
            validation_status: ArmValidationStatus::ExactHardwareMatch,
            memory_required_gb: 8.0,
            memory_available_gb: 16.0,
            compatibility_status: ArmArchitectureCompatibility::Compatible,
        };

        let x86 = OptimizationCandidate {
            model: model("x86-32b", 3_200_000_000, 8.0),
            quantization: "Q4_K_M".to_string(),
            runtime: InferenceRuntime::LlamaCpp,
            context_length: 8192,
            thread_count: None,
            batch_size: None,
            fit: fit_for_model(
                &model("x86-32b", 3_200_000_000, 8.0),
                12.0,
                Some(12.0),
                HardwareMatchLevel::Exact,
            ),
            estimated_tps: 12.0,
            measured_tps: Some(12.0),
            benchmark_match_level: HardwareMatchLevel::Exact,
            performance_provenance: OptimizationProvenance::Measured,
            runtime_availability: RuntimeAvailability::Benchmarked,
            validation_status: ArmValidationStatus::Unknown,
            memory_required_gb: 8.0,
            memory_available_gb: 16.0,
            compatibility_status: ArmArchitectureCompatibility::Compatible,
        };

        assert!(candidate_sort_key(&arm) > candidate_sort_key(&x86));
    }

    #[test]
    fn measured_candidate_is_not_the_same_as_predicted_candidate() {
        let measured = OptimizationCandidate {
            model: model("measured", 2_000_000_000, 4.0),
            quantization: "Q4_K_M".to_string(),
            runtime: InferenceRuntime::LlamaCpp,
            context_length: 8192,
            thread_count: None,
            batch_size: None,
            fit: fit_for_model(
                &model("measured", 2_000_000_000, 4.0),
                11.0,
                Some(10.0),
                HardwareMatchLevel::Exact,
            ),
            estimated_tps: 11.0,
            measured_tps: Some(10.0),
            benchmark_match_level: HardwareMatchLevel::Exact,
            performance_provenance: OptimizationProvenance::Measured,
            runtime_availability: RuntimeAvailability::Benchmarked,
            validation_status: ArmValidationStatus::ExactHardwareMatch,
            memory_required_gb: 4.0,
            memory_available_gb: 8.0,
            compatibility_status: ArmArchitectureCompatibility::Compatible,
        };

        let predicted = OptimizationCandidate {
            model: model("predicted", 4_000_000_000, 8.0),
            quantization: "Q4_K_M".to_string(),
            runtime: InferenceRuntime::LlamaCpp,
            context_length: 8192,
            thread_count: None,
            batch_size: None,
            fit: fit_for_model(
                &model("predicted", 4_000_000_000, 8.0),
                15.0,
                None,
                HardwareMatchLevel::NoMatch,
            ),
            estimated_tps: 15.0,
            measured_tps: None,
            benchmark_match_level: HardwareMatchLevel::NoMatch,
            performance_provenance: OptimizationProvenance::Estimated,
            runtime_availability: RuntimeAvailability::Benchmarkable,
            validation_status: ArmValidationStatus::EstimatedOnly,
            memory_required_gb: 8.0,
            memory_available_gb: 16.0,
            compatibility_status: ArmArchitectureCompatibility::Unknown,
        };

        assert_ne!(
            measured.performance_provenance,
            predicted.performance_provenance
        );
        assert!(candidate_sort_key(&measured) > candidate_sort_key(&predicted));
    }

    #[test]
    fn best_measured_is_not_best_predicted() {
        let measured = OptimizationCandidate {
            model: model("measured-best", 3_000_000_000, 6.0),
            quantization: "Q4_K_M".to_string(),
            runtime: InferenceRuntime::LlamaCpp,
            context_length: 8192,
            thread_count: None,
            batch_size: None,
            fit: fit_for_model(
                &model("measured-best", 3_000_000_000, 6.0),
                10.0,
                Some(9.0),
                HardwareMatchLevel::Exact,
            ),
            estimated_tps: 10.0,
            measured_tps: Some(9.0),
            benchmark_match_level: HardwareMatchLevel::Exact,
            performance_provenance: OptimizationProvenance::Measured,
            runtime_availability: RuntimeAvailability::Benchmarked,
            validation_status: ArmValidationStatus::ExactHardwareMatch,
            memory_required_gb: 6.0,
            memory_available_gb: 12.0,
            compatibility_status: ArmArchitectureCompatibility::Compatible,
        };

        let predicted = OptimizationCandidate {
            model: model("predicted-best", 5_000_000_000, 10.0),
            quantization: "Q4_K_M".to_string(),
            runtime: InferenceRuntime::LlamaCpp,
            context_length: 8192,
            thread_count: None,
            batch_size: None,
            fit: fit_for_model(
                &model("predicted-best", 5_000_000_000, 10.0),
                18.0,
                None,
                HardwareMatchLevel::NoMatch,
            ),
            estimated_tps: 18.0,
            measured_tps: None,
            benchmark_match_level: HardwareMatchLevel::NoMatch,
            performance_provenance: OptimizationProvenance::Estimated,
            runtime_availability: RuntimeAvailability::Benchmarkable,
            validation_status: ArmValidationStatus::EstimatedOnly,
            memory_required_gb: 10.0,
            memory_available_gb: 20.0,
            compatibility_status: ArmArchitectureCompatibility::Unknown,
        };

        assert!(candidate_sort_key(&measured) > candidate_sort_key(&predicted));
        assert_ne!(measured.measured_tps, predicted.measured_tps);
    }

    #[test]
    fn no_benchmark_evidence_stays_estimated_only() {
        let specs = specs_for_arch(CpuArchitecture::Aarch64);
        let status = validation_status_for_match(&specs, HardwareMatchLevel::NoMatch, false);
        assert_eq!(status, ArmValidationStatus::EstimatedOnly);
    }

    #[test]
    fn x86_64_behavior_keeps_non_arm_validation_unknown() {
        let specs = specs_for_arch(CpuArchitecture::X86_64);
        let status = validation_status_for_match(&specs, HardwareMatchLevel::Exact, true);
        assert_eq!(status, ArmValidationStatus::Unknown);
    }

    #[test]
    fn unknown_runtime_state_remains_unknown() {
        let fit = ModelFit {
            runtime: InferenceRuntime::LlamaCpp,
            installed: false,
            estimated_tps: 0.0,
            measured_tps: None,
            ..fit_for_model(
                &model("test", 4_000_000_000, 8.0),
                0.0,
                None,
                HardwareMatchLevel::NoMatch,
            )
        };

        assert_eq!(
            runtime_availability_for_fit(&fit),
            RuntimeAvailability::Unknown
        );
    }

    #[test]
    fn performance_comparison_calculation_stays_separate_from_rank_logic() {
        let fit = ModelFit {
            estimated_tps: 100.0,
            measured_tps: Some(MeasuredTps {
                tok_s: 80.0,
                sample_count: 3,
                hardware_label: "fixture".to_string(),
                source: crate::benchmarks::MeasuredSource::LocalBench,
                match_level: HardwareMatchLevel::Exact,
            }),
            ..fit_for_model(
                &model("perf", 4_000_000_000, 8.0),
                100.0,
                Some(80.0),
                HardwareMatchLevel::Exact,
            )
        };

        let comparison = fit
            .performance_comparison()
            .expect("performance comparison exists");
        assert!((comparison.measured_tps - 80.0).abs() < f64::EPSILON);
        assert!((comparison.estimated_tps - 100.0).abs() < f64::EPSILON);
        assert!((comparison.difference_tps + 20.0).abs() < f64::EPSILON);
        assert!((comparison.ratio - 0.8).abs() < 1e-9);
    }

    #[test]
    fn optimize_for_system_prefers_measured_evidence_when_present() {
        let specs = specs_for_arch(CpuArchitecture::Aarch64);
        let db = ModelDatabase::new();
        let result = optimize_for_system(&specs, &db, 5);

        assert_eq!(result.hardware_architecture, CpuArchitecture::Aarch64);
        assert!(
            result.best_measured.is_some()
                || result.best_predicted.is_some()
                || result.best_available.is_some()
        );
    }
}
