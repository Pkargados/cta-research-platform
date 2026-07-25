from data.sectors import SECTORS, asset_to_sector, sectors_for_universe


def test_sectors_covers_expected_families():
    expected = {"Energy", "PreciousMetals", "IndustrialMetals", "Grains", "Softs",
                "Livestock", "EquityIndex", "Rates", "FX"}
    assert set(SECTORS.keys()) == expected


def test_sofr_and_lumber_deliberately_absent():
    all_members = {a for members in SECTORS.values() for a in members}
    assert "SOFR" not in all_members
    assert "Lumber" not in all_members


def test_industrial_metals_has_exactly_one_member():
    assert SECTORS["IndustrialMetals"] == ["Copper"]


def test_asset_to_sector_is_flat_reverse_lookup():
    mapping = asset_to_sector()
    assert mapping["Corn"] == "Grains"
    assert mapping["Gold"] == "PreciousMetals"
    assert mapping["Copper"] == "IndustrialMetals"
    # Every asset in SECTORS must appear exactly once in the flattened mapping.
    total_members = sum(len(v) for v in SECTORS.values())
    assert len(mapping) == total_members


def test_sectors_for_universe_filters_members_not_in_universe():
    universe = ["Corn", "Wheat", "Gold"]  # Soybeans etc. excluded
    result = sectors_for_universe(universe)
    assert result["Grains"] == ["Corn", "Wheat"]
    assert result["PreciousMetals"] == ["Gold"]


def test_sectors_for_universe_drops_empty_sectors_entirely():
    universe = ["Gold", "Silver"]  # no Grains/Energy/etc. members present
    result = sectors_for_universe(universe)
    assert "Grains" not in result
    assert "Energy" not in result
    assert set(result.keys()) == {"PreciousMetals"}


def test_sectors_for_universe_does_not_repopulate_outside_universe():
    universe = ["Corn"]
    result = sectors_for_universe(universe)
    assert result["Grains"] == ["Corn"]  # not the full Grains list from SECTORS
