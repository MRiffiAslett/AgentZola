# Debug: Inconsistent Sample Counts (5 vs 10)

## Facts
1. **requirement_summary branch**: Always 5 samples per iteration ✓
2. **master branch**: Sometimes 5, sometimes 10 samples per iteration ❌
3. Code for `_create_sampling_params` is IDENTICAL in both branches
4. Config file shows `tests_per_iteration = 5` in both branches  
5. Model default is `tests_per_iteration = 10` in both branches

## The Mystery

If the code and config are the same, why different behavior?

## Possible Explanations

### 1. Different Config File Used on Remote Server
**Check this:**
```bash
# On your remote server where you ran the tests
cd /vol/bitbucket/mtr25/AgentZola/WhiteFox
cat xilo_xla/config/generator.toml | grep tests_per_iteration
```

Did you use a different config file between branches?

### 2. Config Not Being Loaded Properly in Some Cases
Maybe there's a code path where the config isn't fully loaded and falls back to defaults.

**Check your run command:**
- requirement_summary (working): What command did you use?
- master (broken): What command did you use?

### 3. Parallel Execution Bug
Master added `ProcessPoolExecutor` for parallel test execution. Could this be causing state/config sharing issues?

Maybe each worker process is loading config differently?

### 4. vLLM Non-determinism
The random seed changes per iteration. Could this trigger different behavior in vLLM?

## Action Items

Please run these commands on your **remote server**:

```bash
# 1. Check what config was actually used
cd /vol/bitbucket/mtr25/AgentZola/WhiteFox
echo "=== Current config ==="
grep "tests_per" xilo_xla/config/generator.toml

# 2. Check if there are multiple config files
find . -name "*.toml" -type f 2>/dev/null

# 3. Check the actual log from your run
cd logging
grep -i "tests_per_iteration\|SamplingParams\|num_samples" whitefox-llm-gen.log | head -20

# 4. Count actual samples per iteration for one optimization
cd generated-outputs/whitefox_tests/ReshapeReshapeForwarding
for i in {0..14}; do
  count=$(ls -1 ReshapeReshapeForwarding-it${i}-sample*.py 2>/dev/null | wc -l)
  if [ $count -gt 0 ]; then
    echo "it${i}: $count files"
  fi
done
```

This will help us identify if it's a config issue or a code issue.



