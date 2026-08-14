We are building an Arm-first evolution of this project for the Arm AI Optimization Challenge 2026.

Repository:
https://github.com/AlexsJones/llmfit.git

Local project path:
C:\Users\hp\Desktop\Hackathons\armHack\llmfit

Project direction:
llmfit → ArmFit

Core goal:
Turn llmfit into an Arm-first LLM hardware intelligence, recommendation, and benchmarking tool.

IMPORTANT:
Do NOT rewrite the project from scratch.
Do NOT create a separate toy application.
Build incrementally on the existing llmfit architecture.

I am developing on Windows x86_64.
The final target/validation environment will be native Linux ARM64/aarch64.

The project already builds successfully on my Windows machine with:

cargo build -p llmfit

Therefore preserve the existing working build.

==================================================
PHASE 0 — RECONNAISSANCE FIRST
==================================================

Before modifying code, inspect the repository thoroughly.

Focus especially on:

- llmfit-core
- llmfit-tui
- llmfit-desktop
- llmfit-core/src/hardware.rs
- model scoring/recommendation logic
- benchmark implementation
- speed estimation
- runtime/provider integrations
- CLI commands
- TUI rendering
- existing tests
- GitHub Actions / CI
- platform-specific code

Determine:

1. How SystemSpecs is represented.
2. How CPU architecture is detected.
3. How CPU model/name is detected.
4. How CPU cores/threads are detected.
5. How RAM is detected.
6. How GPU information is detected.
7. How models are scored.
8. How speed is estimated.
9. How actual benchmark results are represented.
10. Where the CLI commands are implemented.
11. Where the TUI displays hardware/model information.
12. What ARM support already exists.

IMPORTANT:
Do not assume ARM support is completely absent.
Search the code first.

After inspection, provide a concise implementation plan identifying the exact files/modules that need modification.

Do NOT start a massive refactor.

==================================================
PHASE 1 — ARM64 HARDWARE DETECTION
==================================================

Implement robust Arm architecture detection.

At minimum distinguish:

- aarch64 / ARM64
- x86_64
- unknown

Use Rust/platform APIs or existing project abstractions wherever possible.

Extend the existing hardware abstraction rather than creating a parallel hardware system.

Detect, where reliably available:

- architecture
- CPU vendor
- CPU model
- physical/logical CPU count
- RAM
- Arm CPU capabilities where available

Potential Arm capabilities:

- NEON / ASIMD
- SVE
- SVE2

IMPORTANT:
Never report a capability unless it was actually detected.
Gracefully handle unavailable information.

Do not break Windows x86_64.

Use conditional compilation where appropriate.

==================================================
PHASE 2 — TESTABLE HARDWARE ABSTRACTION
==================================================

Make hardware detection testable without requiring an actual ARM machine.

Create appropriate abstractions/mocks so unit tests can simulate:

Example:

ARM64:
architecture = aarch64
cores = 8
RAM = 16 GB
NEON = true
SVE = false

x86_64:
architecture = x86_64

Tests should verify:

- ARM64 is recognized correctly.
- x86_64 remains recognized correctly.
- unknown architectures are handled safely.
- ARM capabilities are represented correctly.
- missing hardware information does not crash the application.

Do NOT fake real benchmark results.
Mocking hardware metadata for unit tests is fine.

==================================================
PHASE 3 — ARM-AWARE MODEL RECOMMENDATION
==================================================

Extend the existing llmfit scoring/recommendation system.

Do NOT replace the existing scoring system.

Add an Arm-aware layer that can consider:

- architecture compatibility
- available RAM
- model memory requirement
- quantization
- context length
- runtime/provider availability
- actual benchmark evidence
- Arm-specific benchmark evidence

Recommendations must remain explainable.

For example:

Model: Gemma 3 4B
Architecture: ARM64
Memory fit: Excellent
Runtime compatibility: Supported
Measured Arm performance: available/not available
Recommendation: Excellent

Do not invent measured performance.

If no Arm benchmark exists, explicitly say that the value is estimated or unavailable.

==================================================
PHASE 4 — ARM-SPECIFIC BENCHMARK DATA MODEL
==================================================

Inspect the existing benchmark system first.

Extend it rather than creating a duplicate benchmark framework.

Arm benchmark records should be able to represent:

- architecture
- CPU model
- CPU vendor
- core count
- RAM
- model
- quantization
- runtime/provider
- context length
- prompt length
- generation length
- time-to-first-token
- tokens/sec
- total latency
- memory usage
- runtime/software version
- timestamp

Use a clean serializable schema.

Local benchmark results should be saved before optional sharing.

==================================================
PHASE 5 — ESTIMATE VS ACTUAL
==================================================

This is a major ArmFit feature.

For a model where both estimated and measured performance exist, calculate:

Estimated performance
Actual performance
Difference
Percentage error

Example:

Estimated: 10.5 tok/s
Actual: 11.8 tok/s
Difference: +1.3 tok/s
Error: +12.4%

Do not make statistical claims from insufficient data.

Expose this information through the CLI/TUI where appropriate.

==================================================
PHASE 6 — ARM OPTIMIZATION PROFILE
==================================================

Introduce the concept of an optimization profile.

An optimization profile can eventually include:

- model
- quantization
- runtime/provider
- context length
- thread count
- batch size

IMPORTANT:

Only expose configuration options that are actually supported by the selected runtime/provider.

Do NOT implement low-level ARM assembly.
Do NOT write custom SIMD kernels.
Do NOT introduce llama.cpp/ONNX Runtime from scratch.
Do NOT rewrite inference engines.

The first goal is hardware-aware selection and benchmarking.

==================================================
PHASE 7 — ARM DOCTOR REPORT
==================================================

Add or extend the existing doctor functionality.

On an ARM64 machine, the report should clearly show something like:

ArmFit Hardware Report

Architecture: aarch64
CPU: <actual CPU>
Cores: <actual>
Threads: <actual>
RAM: <actual>

NEON: detected/not detected/unknown
SVE: detected/not detected/unknown
SVE2: detected/not detected/unknown

Supported runtimes:
...

On x86 Windows it should clearly identify:

Architecture: x86_64

Do not pretend the Windows development machine is ARM.

==================================================
PHASE 8 — ARM-FOCUSED CLI
==================================================

Preserve existing llmfit CLI commands.

Do not unnecessarily rename the binary yet.

First determine the existing command structure.

Potential future commands:

llmfit doctor
llmfit recommend
llmfit fit
llmfit bench

If useful, introduce an ARM-specific command/subcommand without breaking existing usage.

Examples:

llmfit arm
llmfit arm doctor

But only implement this if it fits naturally into the existing CLI architecture.

==================================================
PHASE 9 — ARM64 CI
==================================================

Add ARM64 CI validation if practical.

Separate:

1. Build tests
2. Unit tests
3. Architecture-specific tests
4. Real performance benchmarks

Do not run unstable performance benchmarks on every CI build.

The important goal is proving that the project can compile/test on native ARM64.

Use GitHub Actions ARM64 runners if supported by the repository/workflow.

==================================================
PHASE 10 — DOCUMENTATION
==================================================

Update documentation only after the implementation is working.

Document:

- What ArmFit is.
- What was added to llmfit.
- ARM64 support.
- Hardware detection.
- Arm-aware model recommendation.
- Benchmark schema.
- Estimate vs actual.
- How to build on Windows.
- How to build on Linux ARM64.
- How to run ARM64 validation.
- How to run real benchmarks.

Clearly distinguish:

ESTIMATED
vs
MEASURED

Never publish fabricated benchmark numbers.

==================================================
HACKATHON POSITIONING
==================================================

The target is:

Track: Cloud AI
Primary theme:
Arm-specific LLM inference optimization / hardware-aware model selection / benchmarking.

The project should demonstrate measurable optimization value.

Do not claim performance improvements until they are actually benchmarked on ARM64.

The final story should be:

"ArmFit helps developers determine which LLM and configuration is best suited for their specific Arm64 machine, then validates those recommendations with real benchmark measurements."

==================================================
DEVELOPMENT CONSTRAINTS
==================================================

Current development environment:

Windows x86_64.

Native ARM64 environment is not currently available locally.

Therefore:

Windows:
- development
- compilation
- unit tests
- mocked ARM hardware tests

ARM64:
- native hardware detection
- ARM-specific integration tests
- real inference benchmarks
- performance measurements

Never fake ARM results.

==================================================
VERIFICATION
==================================================

After each meaningful phase run:

cargo fmt --all
cargo test --workspace
cargo build -p llmfit

Use clippy where practical.

Do not leave compiler errors or broken tests.

Do not modify unrelated functionality.

==================================================
GIT SAFETY
==================================================

Create a dedicated feature branch for this work.

Do not push directly to upstream/original repository.

Keep changes logically separated.

Suggested commits:

1. arm: add architecture abstraction
2. arm: add hardware detection
3. arm: add capability detection
4. arm: add hardware tests
5. arm: add recommendation layer
6. arm: extend benchmark schema
7. arm: add estimate-vs-actual analysis
8. arm: add CLI/TUI support
9. arm: add ARM64 CI
10. docs: document ArmFit

Do not create all commits artificially if the actual changes are better grouped differently.

==================================================
MOST IMPORTANT INSTRUCTION
==================================================

Do NOT blindly implement all phases in one shot.

Start with PHASE 0.

Inspect the repository and report:

1. Current architecture
2. Existing ARM support
3. Relevant files
4. Existing benchmark infrastructure
5. Existing hardware detection
6. Recommended implementation plan
7. Risks
8. What can be tested on Windows
9. What must wait for ARM64

Then stop and wait for approval before making large architectural changes.

The goal is a technically credible, measurable, Arm-first evolution of llmfit — not a superficial fork.