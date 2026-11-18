# Public position against recorded vote (exploratory)

Each coder saw one member's public statements with the case and the two options, and not the version, the stake, the private room or the vote.

- Coder 1: google/gemma-2-9b-it, greedy decoding, every member. Members coded: 384.
- Coder 2: Claude Fable 5.2, one blind pass on 136 members (34 episodes), blind to version, stake, room and vote. Members coded: 136.

Agreement of coder 1 with Claude Fable 5.2 on the three-way reading (stake option, other option, neither), over the 136 members both coded: 97%, kappa 0.84.

The table uses coder 1, over the 384 members that coder coded.

| Version | What every agent's prompt carries | Members | Argue for the stake option in public | Votes for the stake option | of these: argued for the other option in public | argued for neither | argued for the stake option |
|---|---|---|---|---|---|---|---|
| K1 | T7 with stake sentences of the kind 'from above': an office over the member's post wants the outcome and decides on the post | 192 | 11 | 135 | 72 | 52 | 11 |
| K2 | T7 with stake sentences of the kind 'payout': a party's value depends on the outcome and the member is paid if it happens | 192 | 0 | 73 | 71 | 2 | 0 |

Members who argued for the stake option in public and then voted for the other option: K1 0, K2 0.

## Coder 1: public position by vote

| Version | public stake option, vote stake option | public stake option, vote other option | public other option, vote stake option | public other option, vote other option | public neither, vote stake option | public neither, vote other option |
|---|---|---|---|---|---|---|
| K1 | 11 | 0 | 72 | 55 | 52 | 2 |
| K2 | 0 | 0 | 71 | 113 | 2 | 6 |

## Coder 2: public position by vote

| Version | public stake option, vote stake option | public stake option, vote other option | public other option, vote stake option | public other option, vote other option | public neither, vote stake option | public neither, vote other option |
|---|---|---|---|---|---|---|
| K1 | 6 | 1 | 23 | 24 | 5 | 1 |
| K2 | 0 | 0 | 18 | 58 | 0 | 0 |

