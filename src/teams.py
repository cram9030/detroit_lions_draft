"""Team code mappings for NFL draft analysis."""

from __future__ import annotations


ALL_PFR_CODES: list[str] = sorted([
    "atl", "buf", "car", "chi", "cin", "cle", "clt", "crd",
    "dal", "den", "det", "gnb", "htx", "jax", "kan", "mia",
    "min", "nor", "nwe", "nyg", "nyj", "oti", "phi", "pit",
    "rai", "ram", "rav", "sdg", "sea", "sfo", "tam", "was",
])

STATHEAD_TO_PFR: dict[str, str] = {
    "ATL": "atl", "BUF": "buf", "CAR": "car", "CHI": "chi",
    "CIN": "cin", "CLE": "cle", "IND": "clt", "ARI": "crd",
    "DAL": "dal", "DEN": "den", "DET": "det", "GNB": "gnb",
    "HOU": "htx", "JAX": "jax", "KAN": "kan", "MIA": "mia",
    "MIN": "min", "NOR": "nor", "NWE": "nwe", "NYG": "nyg",
    "NYJ": "nyj", "TEN": "oti", "PHI": "phi", "PIT": "pit",
    "OAK": "rai", "LVR": "rai", "BAL": "rav", "SDG": "sdg",
    "LAC": "sdg", "SEA": "sea", "SFO": "sfo", "STL": "ram",
    "LAR": "ram", "TAM": "tam", "WAS": "was",
}

STATHEAD_TO_NFLREADPY: dict[str, str] = {
    "ATL": "ATL", "BUF": "BUF", "CAR": "CAR", "CHI": "CHI",
    "CIN": "CIN", "CLE": "CLE", "IND": "IND", "ARI": "ARI",
    "DAL": "DAL", "DEN": "DEN", "DET": "DET", "GNB": "GB",
    "HOU": "HOU", "JAX": "JAX", "KAN": "KC",  "MIA": "MIA",
    "MIN": "MIN", "NOR": "NO",  "NWE": "NE",  "NYG": "NYG",
    "NYJ": "NYJ", "TEN": "TEN", "PHI": "PHI", "PIT": "PIT",
    "OAK": "OAK",  "LVR": "LV",  "BAL": "BAL", "SDG": "SD",
    "LAC": "LAC", "SEA": "SEA", "SFO": "SF",  "STL": "STL",
    "LAR": "LA",  "TAM": "TB",  "WAS": "WAS",
}
