# Real signal-head crops (regression fixture)

The held-out ("test") split of the traffic-light image set in
https://github.com/shashankkumbhare/traffic-light-classifier (MIT licence,
see LICENSE.md; the folder `yellow` was renamed `amber`). 181 red, 9 amber,
107 green crops of real vertical signal heads, roughly 17×34 to 40×80 pixels.

These are NOT Cape Coast images. They are used only to check that the HSV
reader works on real lamps (over-exposed centres, cyan LED greens, sky in the
crop) and never confuses colours. The defaults in `SignalConfig` were chosen
on the separate "training" split of the same set, not on these files.
