This dataset contains lap-level data for Formula 1 races, where each row represents a single lap completed by a driver.

COLUMNS
Year – Season of the race
Race – Grand Prix name
Driver – Driver code
LapNumber – Lap index within the race
Position – Driver’s position on that lap
LapTime (s) – Lap time in seconds
Stint – Tire stint number
TyreLife – Number of laps on current tire
Normalized_TyreLife – Tire life normalized within stint
Compound_Encoded – Tire compound
LapTime_Delta – Change in lap time from previous lap
Cumulative_Degradation – Accumulated tire performance drop
Position_Change – Position gain/loss compared to previous lap
RaceProgress – Fraction of race completed (0 → 1)
PitStop – Whether the driver pitted on that lap (0/1)
PitNextLap – Target variable: whether the driver will pit next lap (0/1)
