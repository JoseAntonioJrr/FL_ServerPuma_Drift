"""Agendamento compartilhado de Concept Drift entre cliente e servidor.

Mantém um único ponto de verdade para:
- ``parse_drift_clients`` / ``resolve_drift_clients`` (sintaxe 'all', 'none', '0,1', '0-4,7')
- ``phase_for_round(round_num, args, is_drifted_client=True)`` (abrupto, gradual, periódico)
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DriftPhase:
    """Estado de dados esperado para um cliente em uma rodada 1-based."""
    phase: str | None  # None (sem drift), 'phase1', 'phase2' ou 'blend'
    alpha: float = 0.0


def parse_drift_clients(raw, num_clients=None):
    """Converte 'all' | 'none' | '0,1,5' | '0-4,7' em set de IDs PUMA (base 0).

    Se ``raw == 'all'`` e ``num_clients is None``, retorna ``None`` (use ``resolve_drift_clients``).
    """
    raw = (raw or "").strip().lower()
    if raw in ("", "none"):
        return set()
    if raw == "all":
        return set(range(num_clients)) if num_clients is not None else None
    out = set()
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
            lo, hi = int(a), int(b)
            if hi < lo:
                raise ValueError(f"Intervalo inválido em drift_clients: {tok}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(tok))
    if num_clients is not None:
        bad = [c for c in out if not 0 <= c < num_clients]
        if bad:
            raise ValueError(f"drift_clients fora de 0..{num_clients - 1}: {sorted(bad)}")
    return out


def resolve_drift_clients(raw, all_clients):
    """Resolve ``raw`` contra a coleção ``all_clients`` de IDs conhecidos."""
    parsed = parse_drift_clients(raw, num_clients=None)
    if parsed is None:
        return set(all_clients)
    return parsed


def phase_for_round(round_num, args, is_drifted_client=True, legacy_gradual_server_hold=False):
    """Retorna o ``DriftPhase`` esperado na rodada 1-based ``round_num``.

    Compartilhado entre ``client.py`` e ``server.py`` para garantir concordância
    de fase/alpha nos modos abrupto, gradual e periódico.
    """
    if round_num < 1:
        raise ValueError(f"round_num deve ser >= 1 (1-based); recebido {round_num}")

    if not bool(getattr(args, "drift", False)):
        return DriftPhase(None, 0.0)

    if not is_drifted_client:
        return DriftPhase("phase1", 0.0)

    drift_type = getattr(args, "drift_type", "abrupt")
    drift_round = int(getattr(args, "drift_round", 10))
    drift_length = int(getattr(args, "drift_length", 5))
    drift_freq = int(getattr(args, "drift_freq", 0) or 0)

    if drift_freq > 0:
        if drift_type == "gradual":
            L = drift_freq
            p = (round_num - 1) // L
            cycle_step = (round_num - 1) % L
            r = cycle_step / (L - 1) if L > 1 else 0.0
            alpha_interp = r if (p % 2 == 0) else (1.0 - r)
            return DriftPhase("blend", float(alpha_interp))
        if drift_type == "abrupt":
            target = "phase2" if ((round_num // drift_freq) % 2 == 1) else "phase1"
            return DriftPhase(target, 1.0 if target == "phase2" else 0.0)
        raise ValueError(f"drift_type desconhecido: {drift_type}")

    if drift_type == "abrupt":
        if round_num >= drift_round:
            return DriftPhase("phase2", 1.0)
        return DriftPhase("phase1", 0.0)

    if drift_type == "gradual":
        if round_num < drift_round:
            return DriftPhase("phase1", 0.0)
        if round_num < drift_round + drift_length:
            if legacy_gradual_server_hold:
                return DriftPhase("phase1", 0.0)
            alpha_interp = (round_num - drift_round + 1) / (drift_length + 1)
            return DriftPhase("blend", float(alpha_interp))
        return DriftPhase("phase2", 1.0)

    raise ValueError(f"drift_type desconhecido: {drift_type}")

