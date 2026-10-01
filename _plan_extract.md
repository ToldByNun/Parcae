# Theory CUDA Throughput: Diagnose → Transpiler → 90% Peak

## Acceptance (verbindlich umgeschrieben)

**PRIMARY (immer):** `measured ≥ 0.90 × estimated_peak(theory_shape)`  
Metrik: cudaEvent median-of-3, setup excluded, T≥2^20. Peak-Dokumentation Pflicht.

**50B:** nur optionaler Zwischen-Checkpoint wenn `estimated_peak ≫ 50B` — **kein** Done-Kriterium und **kein** Ersatz für 90%.

```text
done ⇔ measured_runes_per_s >= 0.90 * estimated_peak(theory_shape)
# optional interim only if peak >> 50B:
checkpoint_50B ⇔ measured >= 50e9   # allein = NICHT done
```

Beispiele:
- `bitmask_blend` S2, Peak 350B → Done-Gate **315B** (50B nur „unterwegs“)
- Autokey-schwer, Peak 80B → Done-Gate **72B** (50B nie das Ziel)

Plan-Datei: [theory_cuda_throughput_climb_1eea74f6.plan.md](c:\Users\mikaj\.cursor\plans\theory_cuda_throughput_climb_1eea74f6.plan.md) (bereits aktualisiert).

## Unverändert (Rest des Plans)

Diagnose-Schichten A–E, nsys/ncu-Playbook, Host-Amortisierung, Interpreter Quick Wins, `TheoryHistChi2Emit` S1/S2 AOT, Export prefer specialized, Commit-Checkliste — wie zuvor; nur Gates/Erfolgskriterien/Block-4/Commit-13 auf **90%-Peak-primary** umgestellt.

Agent: nur code+test; kein git/gh.