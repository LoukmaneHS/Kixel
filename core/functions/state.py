def state(karacter: Karacter) -> dict[str, np.ndarray]:
    return {
        "nacc": karacter.nacc.copy(),
        "oacc": karacter.oacc.copy(),
        "sacc": karacter.sacc.copy(),
    }
