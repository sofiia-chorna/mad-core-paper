from mad.madcore_vs_random_20k.random_sampler import run_sampler

if __name__ == "__main__":
    run_sampler("random_test_6k", 6000, 20260814,
                ["out/madcore_25k/selection.tsv", "out/random_25k/selection.tsv"])
