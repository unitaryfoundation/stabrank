"""Frozen proposed portable parameters; no previously generated scientific data."""
PARAMETERS = {
    'seconds': 3600, 'preflight_seconds': 60, 'workers': 4, 'case_batch': 32, 'inflight_batches': 8,
    'transition_cap': 1024, 'ordered_pair_cap': 65536,
    'batch_identity_cap': 100000, 'batch_identity_byte_cap': 33554432,
    'unique_cover_cap': 100000, 'prime': 65521, 'imaginary_unit': 41224,
    'expected_dictionary_rays': 36720, 'expected_minimum_three_covers': 16,
    'expected_minimum_three_orbits': 1,
    'projection': [
      [8437,19617,29453,8550,34095,45590,43959,51347,10289,21749,63453,43097,1420,42060,20526,30675],
      [62972,20081,23495,1621,16366,9035,64532,62075,24983,39723,59362,3432,50585,11652,37244,7418],
      [9457,64886,35301,47371,24605,61397,55136,37946,4853,9927,33853,49657,56778,10062,19776,43532]],
    'use_product_family_filter': False, 'use_one_entangled_family_filter': False,
    'use_fidelity_filter': False, 'reuse_precomputed_dictionary_or_cover_data': False,
}
