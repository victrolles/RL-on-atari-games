I'd consider these improvements after verifying the implementation:
- Frame skipping: Keep actions for two or three consecutive simulation frames, reducing neural-network inference frequency. Implement this with a wrapper that also accumulates rewards and stops immediately on termination.
- Prioritized or improved replay storage: Especially useful once you move beyond 50,000 transitions.
- Image augmentation: Random shifts/crops can improve robustness. DrQ-style SAC is a particularly relevant next step.
- Reward diagnostics: Log average speed, steering, gas, brake, episode length, and reward—not just actor and critic losses.
- Evaluate on unseen tracks: CarRacing generates a new track every episode, so evaluation across several random tracks is more informative than one successful run.