# Stage 2: what the rules do when there is nothing to find

100 passes of the probe sweep in which every label was first shuffled inside its matched set. The true score is 0.5 everywhere, so whatever fires here fires by chance. The same activations, folds, score and intervals as the real sweep.

| Contrast | Passes | Signal at the end of the room (Q1) | An onset somewhere (Q2) | An anchor inside the room (15.2) | More than words at some position (Q4) | Positions with an interval above 0.5 | Best position of a timeline: median, 95th percentile, highest | Mean over the room positions: 95th percentile; passes with its interval above 0.5; passes 'more than words' in the room | Score at the end of the room: median, 95th percentile, highest | Best position inside the room: median, 95th percentile, highest |
|---|---|---|---|---|---|---|---|---|---|---|
| vote | 100 | 2 | 0 | 3 | 0 | 184 of 3200 | 0.594, 0.646, 0.667 | 0.524; 7; 0 | 0.506, 0.572, 0.610 | 0.594, 0.646, 0.667 |

Reading: a rule that fires in more than about one pass in twenty is not evidence by itself. The real score at the end of the room counts as a signal only above the 95th percentile of the column 'Score at the end of the room', and an onset counts only if the best position of the real timeline is above the 95th percentile of the column 'Best position of a timeline'.
