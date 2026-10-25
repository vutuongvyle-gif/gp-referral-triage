from referral_triage.privacy.masking import build_analyzer, mask, unmask

NOTE = ("Referral for Aroha Ngata (NHI ZGM93JM), DOB 14/03/1961, ph 021 555 0123, "
        "aroha@example.com. Ms Ngata reports 8 months of knee pain.")


def test_mask_and_restore():
    masked, tokens = mask(NOTE, build_analyzer(), known_names=["Aroha Ngata", "Ngata"])
    assert "ZGM93JM" not in masked and "Ngata" not in masked and "021 555" not in masked
    assert unmask(masked, tokens) == NOTE