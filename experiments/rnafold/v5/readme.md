# Our Scallop Methodology

```
Strategy 1:
- train:     [x] -> nn{theta} -> [r^] -> scallop -> [y^]==[y]
- inference: [x] -> nn{theta} -> [r^] -> scallop -> [y^]==[y]

Strategy 2:
                                     /==[r] <- scallop_back <- [y]
- train:     [x] -> nn{theta} -> [r^]
                                     \-> scallop_forward -> [y^]==[y]
- inference: [x] -> nn{theta} -> [r^] -> scallop_forward -> [y^]
```
