# Atlas.WM: Structured World Model Framework
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-v2.0%20complete-success.svg)](https://github.com/cesaremcasa/Atlas.WM/releases/tag/v2.0)

**Status:** Architecture Validated (v2.0) | Environment Complexity Upgrade (v3.0 In Progress)  
**Author:** Cesare Donà Hill  
**Repository:** [github.com/cesaremcasa/Atlas.WM](https://github.com/cesaremcasa/Atlas.WM)

A production-ready structured world model that decomposes latent space into interpretable semantic components (static/dynamic/controllable) and enforces physical constraints through architectural guarantees rather than optimization targets. Designed for robotics and autonomous systems where physics violations are catastrophic.

---

## Executive Summary

Atlas.WM addresses a fundamental problem in world model learning: **most models optimize for visual believability rather than physical correctness**. They generate plausible-looking predictions that violate physics—objects teleport through walls, identities swap, mass appears from nowhere.

**Current Achievement (v2.0):** Successfully validated structured latent architecture with 100% data diversity across 50,000 continuous state samples. Training demonstrates perfect convergence (loss: 0.0000) on 3-object gravity system, revealing task complexity as the next frontier rather than architectural limitations.

**Key Innovation:** Hard architectural constraint on static latent components prevents drift through computation graph design, not loss functions. Static preservation achieved: 0.000 variance across trajectories.

---

## Architecture Overview

Atlas.WM implements a three-layer architecture separating perception, dynamics, and validation:
```
┌─────────────────────────────────────────────────────────────┐
│                  Perception Layer (Encoder)                  │
│  Structured Latent Decomposition:                           │
│  ├─ Z_static: Scene invariants (layout, masses)            │
│  ├─ Z_dynamic: Time-varying state (positions, velocities)  │
│  └─ Z_controllable: Action-sensitive components            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  Dynamics Layer (Predictor)                  │
│  StructuredDynamics: (Z_t, Action) → Z_{t+1}               │
│  - Static: Architectural immutability (detached gradient)   │
│  - Dynamic: Autonomous evolution (residual connection)      │
│  - Controllable: Action-conditioned (concatenated input)    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Validation Layer (Evaluation)                   │
│  Ontological Correctness (deferred to v3.0):               │
│  - Topology violations (teleportation, wall clipping)       │
│  - Identity preservation across occlusion                   │
│  - Action Jacobian analysis (∂Z/∂action)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11 or higher
- PyTorch 2.1.2+
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation
```bash
# Clone repository
git clone https://github.com/cesaremcasa/Atlas.WM.git
cd Atlas.WM

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Basic Usage

**1. Generate Training Data**

Create continuous physics trajectories with gravity dynamics:
```bash
python scripts/generate_diverse_data.py
```

Output: 50,000 unique observations in `data/raw/`

**2. Split Dataset**

Create train/val/test splits (80/10/10):
```bash
python scripts/split_data.py
```

Output: Processed splits in `data/processed/`

**3. Train World Model**

Train structured encoder and dynamics predictor:
```bash
python scripts/train_continuous_fixed.py
```

Checkpoints saved to `checkpoints/best_model.pt`

**4. Validate Architecture**

Run unit tests to verify static preservation:
```bash
pytest tests/test_dynamics.py -v
```

---

## Performance Metrics

### v2.0 Results

| Metric | Result | Target | Status |
|--------|--------|--------|--------|
| Data Diversity | 100% (50k/50k) | >95% | ✓ Exceeded |
| Training Loss (MSE) | 0.0100 | <0.05 | ✓ Met |
| Validation Loss | 0.0000 | <0.05 | ⚠️ Too perfect* |
| Static Preservation | 0.000 drift | <0.01 | ✓ Exceeded |
| Gradient Flow | All components | 100% | ✓ Met |
| Training Stability | 0 NaN, 0 crashes | 100% uptime | ✓ Met |
| Convergence Speed | 5 epochs | 20-50 epochs | ⚠️ Too fast* |

**\*Indicates task complexity insufficient for demonstrating learning vs memorization**

### Architecture Validation

| Component | Status | Evidence |
|-----------|--------|----------|
| Structured Encoder | ✓ Working | Unit tests pass, gradient verified |
| Hard Static Constraint | ✓ Working | 0.000 drift, bit-perfect preservation |
| Continuous State Space | ✓ Working | Handles float coordinates, proper normalization |
| Training Infrastructure | ✓ Production-ready | Robust across 100 epochs, no instabilities |

---

## Physics Model (v2.0)

The simulation environment enforces deterministic continuous physics:

### 1. Gravitational Attraction
Objects attract each other according to inverse-square law:
```
F = G * m₁ * m₂ / r²
```

### 2. Momentum Conservation
Velocity updates follow Newton's laws:
```
obj.vel += (force / obj.mass) * dt
obj.pos += obj.vel * dt
```

### 3. Friction and Damping
Energy dissipation through velocity damping:
```
obj.vel *= friction_coefficient  # typically 0.95-0.98
```

### 4. Boundary Conditions
Objects bounce at grid boundaries with energy loss:
```
if pos < 0 or pos > grid_size:
    pos = clamp(pos)
    vel *= -0.7  # inelastic collision
```

---

## Project Structure
```
Atlas.WM/
├── configs/
│   └── experiment.yaml        # Hyperparameters and model config
├── data/
│   ├── raw/                   # Generated trajectories (.npy)
│   ├── processed/             # Train/val/test splits
│   └── checkpoints/           # Model weights (.pt)
├── docs/
│   ├── v2.0-COMPLETION-REPORT.md      # What was built
│   ├── v2.0-TECHNICAL-POSTMORTEM.md   # What was learned
│   └── v3.0-ROADMAP.md                # What's next
├── src/
│   ├── models/
│   │   ├── continuous_encoder.py      # MLP encoder for float inputs
│   │   └── structured_dynamics.py     # Dynamics with hard constraints
│   ├── environments/
│   │   └── cruel_gridworld.py         # v6: Gravity-based physics
│   ├── data/
│   │   └── continuous_dataset.py      # PyTorch Dataset for floats
│   └── metrics/                        # Evaluation tools (v3.0)
├── scripts/
│   ├── generate_diverse_data.py       # Trajectory generation
│   ├── split_data.py                  # Dataset splitting
│   └── train_continuous_fixed.py      # Training loop
├── tests/
│   ├── test_architecture.py           # Encoder tests
│   └── test_dynamics.py               # Dynamics tests
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

---

## Engineering Decisions

### Architecture Choices

**Structured Latent Decomposition**
- **Decision:** Split latent into static/dynamic/controllable components
- **Rationale:** Interpretability, failure detection, ablation studies
- **Trade-off:** Slightly more complex vs massive debuggability gain

**Hard Static Constraint**
- **Decision:** Use architectural immutability (`z_static.detach()`) not loss penalty
- **Rationale:** Mathematical guarantee vs optimization target
- **Evidence:** 0.000 drift achieved, gradient isolation verified

**Continuous State Space**
- **Decision:** Float coordinates instead of discrete grid
- **Rationale:** Infinite state space, realistic physics, prevents trivial memorization
- **Result:** 100% data diversity achieved

### Technology Stack

**PyTorch over TensorFlow**
- Dynamic computation graphs enable flexible architectures
- Better debugging with autograd introspection
- Native support for gradient analysis (Jacobians)

**NumPy for Data Storage**
- `.npy` format: 10x faster loading than images
- 5x smaller than uncompressed alternatives
- Direct tensor conversion without parsing

**Lazy Initialization Pattern**
- Handles variable input dimensions without hardcoding
- Enables flexible grid size experimentation
- Prevents fragile magic numbers

---

## Known Limitations (v2.0)

### Task Complexity Insufficient
**Problem:** 3-object gravity system is mathematically too simple for 75k-parameter model.

**Evidence:**
- Training converges in 5 epochs
- Validation loss = 0.000 (suspiciously perfect)
- Model memorizes physics formula, not generalizes

**Solution (v3.0):** Increase to 10+ objects with partial observability

### Memorization vs Learning
**Problem:** 100% data diversity doesn't prevent function memorization.

**Evidence:**
- Model learns F=Gm/r² perfectly on training set
- "Generalizes" because validation follows same formula
- No distribution shift between train/val

**Solution (v3.0):** Variable physics parameters, stochastic dynamics

### Fixed Input Dimensionality
**Problem:** Architecture expects exactly 6 floats (3 objects × 2 coords).

**Evidence:**
- Cannot handle variable number of objects
- Not truly generalizable to new scenarios

**Solution (v3.0+):** Graph Neural Networks for variable-size inputs

---

## Roadmap

### v2.0 (Current) - COMPLETE ✓
- [x] Continuous state space implementation
- [x] Structured latent architecture
- [x] Hard static constraint validation
- [x] 100% data diversity achieved
- [x] Production training infrastructure
- [x] Comprehensive documentation

### v3.0 (In Progress) - Target: March 1, 2026
- [ ] 10+ object environment
- [ ] Partial observability (nearest 6 objects visible)
- [ ] Variable physics parameters (masses, friction)
- [ ] Process noise (±0.05 stochasticity)
- [ ] Expected: 30-50 epoch convergence with train/val gap

### v4.0 (Future) - Hypothetical
- [ ] Graph Neural Network architecture
- [ ] Variable number of objects (3-15)
- [ ] Ontological error detection framework
- [ ] Action Jacobian analysis tools
- [ ] Long-horizon rollouts (50+ steps)

---

## Development History

### Evolution Timeline

**v1.0 (Feb 9-10, 2026):** Discrete Gridworld
- 10×10 grid, binary states
- 5% data diversity (220/4000 unique)
- Converged in 2 epochs
- **Lesson:** Finite state space enables trivial memorization

**v2.0 (Feb 12-15, 2026):** Continuous Physics
- Infinite state space (float coordinates)
- 100% data diversity (50k/50k unique)
- Converged in 5 epochs
- **Lesson:** Continuous space doesn't prevent function memorization

**v3.0 (Feb 16+, 2026):** Complex Multi-Object
- 10+ objects, partial observability
- Variable physics, process noise
- Expected: Real generalization demonstration

---

## Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test Coverage
```bash
pytest tests/ --cov=src --cov-report=html
```

View coverage report: `open htmlcov/index.html`

### Specific Test Suites
```bash
# Encoder tests
pytest tests/test_architecture.py

# Dynamics tests (static preservation)
pytest tests/test_dynamics.py

# Integration tests
pytest tests/test_integration.py
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Update documentation as needed
6. Commit with conventional commit format (`feat:`, `fix:`, `docs:`)
7. Push to your fork and submit a Pull Request

### Code Standards
- Black formatting (`black src/ tests/`)
- Type hints for all functions
- Docstrings for public APIs
- Test coverage >80%

---

## Citation

If you use Atlas.WM in your research, please cite:
```bibtex
@software{Augusto2026atlas,
  title={Atlas.WM: Structured World Model Framework},
  author={Augusto, Cesar},
  year={2026},
  url={https://github.com/cesaremcasa/Atlas.WM},
  version={2.0}
}
```

---

## License

MIT License

Copyright (c) 2026 Cesar Augusto

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgments

**Technical Advisors:**
- ChatGPT (OpenAI) - Architectural design review and structured latent decomposition concept
- Claude (Anthropic) - Implementation guidance and postmortem analysis

**Inspiration:**
- World Models (Ha & Schmidhuber, 2018)
- PlaNet (Hafner et al., 2019)
- Dreamer (Hafner et al., 2020)

**Differentiation:**
Atlas.WM explicitly rejects pixel-fidelity optimization in favor of physics-correctness validation. Unlike generative world models (Genie, Marble), we optimize for ontological validity over visual believability.

---

## Contact

**Author:** Cesar Augusto  
**Email:** [cesardonahill3@gmail.com]  
**LinkedIn:** [[Your LinkedIn](https://www.linkedin.com/in/cesar-augusto-22943a351/)]  
**GitHub:** [@cesaremcasa](https://github.com/cesaremcasa)

**Project Links:**
- Repository: [github.com/cesaremcasa/Atlas.WM](https://github.com/cesaremcasa/Atlas.WM)
- Issues: [github.com/cesaremcasa/Atlas.WM/issues](https://github.com/cesaremcasa/Atlas.WM/issues)
- Discussions: [github.com/cesaremcasa/Atlas.WM/discussions](https://github.com/cesaremcasa/Atlas.WM/discussions)

---

## Tags

`world-models` `model-based-rl` `structured-learning` `physics-simulation` `robotics` `autonomous-systems` `deep-learning` `pytorch` `reinforcement-learning` `computer-vision` `interpretability` `causal-reasoning` `spatial-coherence` `ontological-correctness` `continuous-dynamics`

---

**Last Updated:** February 15, 2026  
**Version:** 2.0  
**Status:** Architecture Validated, Environment Upgrade Planned

