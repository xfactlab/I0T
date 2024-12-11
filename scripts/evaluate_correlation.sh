#!/bin/bash

python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model our_longclip --compute_refpac
python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model our_longclip --compute_refpac

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model clip --compute_refpac
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model clip --compute_refpac

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model clip --compute_refpac --random_init t
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model clip --compute_refpac --random_init t

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model clip --compute_refpac --shift_features t
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model clip --compute_refpac --shift_features t

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model clip --compute_refpac --shift_features t --delta 1
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model clip --compute_refpac --shift_features t --delta 1

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model pacscore --compute_refpac
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model pacscore --compute_refpac

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model longclip --compute_refpac
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model longclip --compute_refpac

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model longclip --compute_refpac --remove_artifacts t
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model longclip --compute_refpac --remove_artifacts t

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model clipcloob --compute_refpac
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model clipcloob --compute_refpac

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model cyclip --compute_refpac
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model cyclip --compute_refpac

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model blip --compute_refpac
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model blip --compute_refpac

# python3 ../evaluation/compute_correlations.py --datasets flickr8k_cf --clip_model blip2 --compute_refpac
# python3 ../evaluation/compute_correlations.py --datasets flickr8k_expert --clip_model blip2 --compute_refpac