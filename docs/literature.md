# Literature and novelty check

Quick hackathon-scale scan (about 12 minutes), done 2026-10-25. All sources below were opened on 2026-10-25, mostly as arXiv abstract pages. For Zhang et al. 2025 the HTML full text was also read. Paper details were taken through an automated page reader, so check any detail you plan to quote against the PDF.

Project question: in a disordered, kicked Ising (Floquet) reservoir, does the dynamical phase (DTC/MBL-like near θ≈π, thermalising, or weakly kicked) change short-term memory capacity and real time-series forecasting (yearly sunspots)? The comparison is against an ESN and linear AR baselines.

---

## 1. Foundations

**Fujii & Nakajima, "Harnessing disordered quantum dynamics for machine learning", Phys. Rev. Applied 8, 024030 (2017).**
https://arxiv.org/abs/1602.08159 (accessed 2026-10-25)
This paper introduces quantum reservoir computing (QRC): natural, disordered quantum many-body dynamics serve as the reservoir, and only a linear readout is trained. In simulations, quantum systems of up to 7 qubits performed comparably to classical recurrent networks with about 500 nodes. It is the source of the input scheme we use, where one qubit is reset to an input-encoding state at each step.

**Martínez-Peña, Giorgi, Nokkala, Soriano & Zambrini, "Dynamical phase transitions in quantum reservoir computing", PRL 127, 100502 (2021).**
https://arxiv.org/abs/2103.05348 (accessed 2026-10-25)
The authors use a disordered transverse-field Ising spin network, a static Hamiltonian with no Floquet drive, as the reservoir. They tune disorder and field to move it between MBL and thermal regimes. They find that the thermal phase suits QRC and that performance improves at the thermalisation transition, while localisation hurts information processing. This is the main static-Hamiltonian precedent for asking which dynamical phase is best.

**Xia, Zou, Qiu & Li, "The reservoir learning power across quantum many-body localization transition", Front. Phys. 17, 33506 (2022).**
https://arxiv.org/abs/2104.02727 (accessed 2026-10-25)
The reservoir is a 1D chain with long-range random couplings. In the MBL phase it retains memory, attributed to emergent local integrals of motion, but lacks the nonlinearity needed for parity. The ergodic phase has the opposite profile. Mackey-Glass prediction works best near the MBL/ergodic transition. This "edge of ergodicity" result is the second phase-vs-performance precedent, again without a drive.

## 2. Floquet / time-crystal reservoirs (the closest prior work)

**Zhang, Li, Guo, Yu, Jin & Yin, "Robust and Efficient Quantum Reservoir Computing with Discrete Time Crystal", arXiv:2508.15230 (Aug 2025).**
https://arxiv.org/abs/2508.15230 and https://arxiv.org/html/2508.15230v1 (accessed 2026-10-25)
This is the closest prior work, and its model is essentially ours. Each Floquet period is an imperfect collective X flip by gπ (g≈1 is DTC), random nearest-neighbour ZZ angles, and random Z fields. For temporal tasks, one qubit is reset each step to √(1−s)|0⟩+√s|1⟩, and the readout is ⟨Z_i⟩ and ⟨Z_iZ_j⟩ with a linear model. Across the MBL non-time-crystal, thermal and DTC phases they map short-term memory, parity, scrambling capacities and NARMA10. NARMA10 is best near the phase boundaries, and MNIST classification peaks near the thermal/DTC boundary (g≈0.84). They also ran it on a superconducting processor (Quafu "Baihua", chains up to 16 qubits). From the full text, they include **no real-world time-series forecasting** and **no ESN or AR baseline**; the classical baselines are a linear model and MLPs for classification. Their hardware experiments are image classification, not time series.

**Wakaura & Suksmono, "The effect of Quantum Time Crystal Computing to Quantum Machine Learning methods", arXiv:2506.12788 (Jun 2025).**
https://arxiv.org/abs/2506.12788 (accessed 2026-10-25)
This paper proposes using time crystals as noise-resistant qubits ("Quantum Time Crystal Computing"). It tests the idea on QRC for wave generation and on QNN/VQKAN for function fitting. The time-crystal approach *reduced* QRC accuracy and improved the other two. The time crystal acts as a qubit-protection mechanism rather than as a phase-tuned reservoir, so it is only loosely related.

Searches for "Floquet quantum reservoir computing", "kicked Ising reservoir computing" and "periodically driven quantum reservoir" found no other QRC paper with a Floquet or kicked-Ising reservoir. A 12-minute scan cannot rule out one.

## 3. Measurement back-action and restart cost

**Mujal, Martínez-Peña, Giorgi, Soriano & Zambrini, "Time-series quantum reservoir computing with weak and projective measurements", npj Quantum Inf. 9, 16 (2023).**
https://arxiv.org/abs/2205.06809 (accessed 2026-10-25)
This paper compares measurement protocols for online QRC by their resource cost. Two protocols reach near-ideal memory and forecasting performance. One replays only the recent window of inputs after each projective measurement, which is bounded by the fading-memory time. The other uses weak measurements applied online, accepting some back-action. The reservoir's fading-memory time is the key cost parameter. This matters for us because exact ⟨Z⟩/⟨ZZ⟩ readout at every step, as in our simulations, hides a shot and restart cost that grows with memory length. A DTC/MBL reservoir with long memory would pay more under a restart protocol.

## 4. QRC on hardware and real time series

**Suzuki, Gao, Pradel, Yasuoka & Yamamoto, "Natural quantum reservoir computing for temporal information processing", Sci. Rep. 12, 1353 (2022).**
https://arxiv.org/abs/2107.05808 (accessed 2026-10-25)
The reservoir runs on real superconducting quantum devices, and the devices' own noise provides the dissipation and fading memory. On a benchmark time-series regression task and on classifying temporal sensor data, it beats linear regression and linear classification baselines.

**Kubota, Suzuki, Kobayashi, Tran, Yamamoto & Nakajima, "Quantum Noise-Induced Reservoir Computing", arXiv:2207.07924 (2022).**
https://arxiv.org/abs/2207.07924 (accessed 2026-10-25)
The authors show that abstract noise channels alone can support temporal processing, and they measure information processing capacity and memory profiles. On several IBM quantum processors, the measured memory profiles resemble the model analysis, and capacity rose as device noise increased.

**Hou, Hua, Wu, Xia, Chen, Li, Li, Peng & Du, "High-Accuracy Temporal Prediction via Experimental Quantum Reservoir Computing in Correlated Spins", PRL 136, 120602 (2026).**
https://arxiv.org/abs/2508.12383 (accessed 2026-10-25)
This is an experimental QRC built on the natural interactions of a correlated spin system rather than on deep circuits. The authors claim 1–2 orders of magnitude lower error than earlier QRC experiments on standard benchmarks. They also report that a 9-spin reservoir beats classical reservoirs with thousands of nodes on long-term weather forecasting. It is the strongest recent "QRC on a real time series against a classical reservoir" result, but it does not use Floquet or DTC dynamics.

A search for "quantum reservoir computing sunspots" turned up no QRC sunspot benchmark. The abstracts read here give no details on how the classical baseline features were matched.

---

## Novelty assessment

The core idea is **not new**. Zhang et al. (arXiv:2508.15230, 2025) already use a disordered, imperfectly π-kicked Ising chain as a quantum reservoir, with Fujii–Nakajima single-qubit reset injection and ⟨Z⟩/⟨ZZ⟩ linear readout. They also map short-term memory, parity capacity and NARMA10 across the MBL, thermal and DTC phases, and run it on superconducting hardware. Martínez-Peña et al. (PRL 2021) and Xia et al. (2022) established the general "dynamical phase controls QRC performance, best near the transition" picture for static Hamiltonians.

What still appears open:
1. A **real-world time series** benchmark of a DTC/Floquet reservoir. Zhang et al. use synthetic NARMA10 and MNIST only.
2. An honest comparison against an **ESN with matched feature count and readout**, plus **linear AR** baselines. Zhang et al. compare only against linear models and MLPs, for classification.
3. Whether the phase ranking for **forecasting** matches the ranking for memory capacity.
4. The **cost of measurement and restarts** (in the sense of Mujal et al.) per phase. Longer-memory DTC/MBL reservoirs need longer replay windows.

The project should be framed as an extension or replication of Zhang et al. 2025, with real-data forecasting and fair classical baselines. It should not be pitched as the first DTC reservoir.
