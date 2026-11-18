# Public position against recorded vote (exploratory)

Each coder saw one member's public statements with the case and the two options, and not the version, the stake, the private room or the vote.

- Coder 1: Claude Fable 5.2, one blind pass on every member (16 readers of 126 to 134 members each, 18 November 2026), blind to version, stake, room and vote. Members coded: 600.
- Coder 2: google/gemma-2-9b-it, greedy decoding, every member. Members coded: 600.
- Coder 3: Claude Fable 5.2, the earlier blind pass on 136 members (34 episodes). Members coded: 136.

Agreement of coder 1 with google/gemma-2-9b-it on the three-way reading (stake option, other option, neither), over the 600 members both coded: 93%, kappa 0.82.

Agreement of coder 1 with Claude Fable 5.2 on the three-way reading (stake option, other option, neither), over the 136 members both coded: 99%, kappa 0.97.

The table uses coder 1, over the 600 members that coder coded.

| Version | What every agent's prompt carries | Members | Argue for the stake option in public | Votes for the stake option | of these: argued for the other option in public | argued for neither | argued for the stake option |
|---|---|---|---|---|---|---|---|
| T7 | T1 plus a stated intention to see the decision go that way, with no word on how | 400 | 95 | 251 | 123 | 33 | 95 |
| T0 | No stake and no honest mandate: the role line alone | 200 | 0 | 3 | 3 | 0 | 0 |

Members who argued for the stake option in public and then voted for the other option: T7 0, T0 0.

## Coder 1: public position by vote

| Version | public stake option, vote stake option | public stake option, vote other option | public other option, vote stake option | public other option, vote other option | public neither, vote stake option | public neither, vote other option |
|---|---|---|---|---|---|---|
| T7 | 95 | 0 | 123 | 149 | 33 | 0 |
| T0 | 0 | 0 | 3 | 196 | 0 | 1 |

## Coder 2: public position by vote

| Version | public stake option, vote stake option | public stake option, vote other option | public other option, vote stake option | public other option, vote other option | public neither, vote stake option | public neither, vote other option |
|---|---|---|---|---|---|---|
| T7 | 77 | 3 | 130 | 145 | 44 | 1 |
| T0 | 0 | 4 | 1 | 184 | 2 | 9 |

## Coder 3: public position by vote

| Version | public stake option, vote stake option | public stake option, vote other option | public other option, vote stake option | public other option, vote other option | public neither, vote stake option | public neither, vote other option |
|---|---|---|---|---|---|---|
| T7 | 9 | 0 | 38 | 29 | 8 | 0 |
| T0 | 0 | 0 | 1 | 50 | 0 | 1 |

