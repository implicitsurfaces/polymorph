## Polymorph experiments.

Uses Fidget to render SDF to a bitmap, calculate metrics from that (e.g., area), and optimize.


See how pixel counting measurements compare with exact value:

    cargo run --release --bin circle_measurement

Try using a black box optimzer to guess a radius value:

    cargo run --release --bin circle_measurement


This seems to work, but I don't understand why the optimal parameter isn't in the return value:

    Attempting to guess a radius of 0.22
                estimate       relative error
                   0.031               -0.857
                   0.038               -0.828
                   0.045               -0.794
                   0.053               -0.758
                   0.071               -0.678
                   0.113               -0.485
                   0.166               -0.244
                   0.229                0.042
                   0.302                0.373
                   0.264                0.200
                   0.212               -0.035
                   0.204               -0.071
                   0.217               -0.016
                   0.225                0.023
                   0.212               -0.035
                   0.219               -0.006
                   0.223                0.014
                   0.217               -0.016
                   0.220               -0.002
                   0.221                0.005
                   0.219               -0.006
                   0.220                0.000
                   0.220                0.002
                   0.220               -0.001
    (FtolReached, [0.26468749999999996], 0.0004910555752840859)
