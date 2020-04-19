"""
    Firefly.Metrics

This module provides code for analyzing the results from ADI in a way that is interpretable statistically. Some of the key functionalities are signal-to-noise, significance, the receiver operating characteristic, and the contrast curve.
"""
module Metrics

export snr,
       snr_approx,
       snr_approx!,
       snrmap,
       significance

include("snr.jl")

end
