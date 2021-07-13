# Experimental Configs
- nominal reg is the control
- neural structured learning reg, gaussian (noise) reg, and noise/data/blur corruption reg set


# Requirement
- Need to define each exp_config in terms of the strategy, adversarial regularization technique (method name), and then each `.sh` file will execute iteratively in terms of increasing `adv_step_size`
- Need to setup the code inside each client to get and write specific variables into a logfile given `if '__name__' == '__main__'` instance. The code for the plots are defined in `metrics.py`, and the file to analyze the log data and create the plots will be used in `analysis.py`.

# Opt.
- We can measure for `l2` norm as well

## Reference
- Change the variables that will iterate. Set it up for one exp config then all you have to do is change the model specification and the regularization technique
- Use `python3 ../../server.py`

