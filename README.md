# MA-CBS with Joint A* and Nested CBS
This repository contains our implementation of Meta-Agent Conflict-Based Search with two alternative low-level solvers:

- **Joint-State A\***
- **Nested CBS** 

The code includes scripts to generate MAPF instances, run batched experiments, and exports statistics to 'testData.csv' for plotting.

All scripts are intended to be run from the repository root. 

## Generating Random MAPF Instances
Instance generation is handled by the instanceScript.bash (which wraps code/generateProblem.py) and verifies that CBS can solve each instance.
./instanceScript.bash <#agents> <sizeX> <sizeY> <density> <#problems> <timeout>
- <#agents>: The number of agents in each instance.
- <sizeX>, <sizeY>: The grid dimensions.
- <density>: Static obstacle density (0.3 for 30%).
- <#problems>: How many instances to generate.
- <timeout>: Verification timeout (in seconds) for CBS on each instance.
By default the script
- writes tempory instances into tmpInstances/exp<i>.txt
- verifies each instance using CBS
copies valid instances into instances.text<i>.txt

## Running MA-CBS Experiments
Performance experiments are driven testSuite.bash, which repeatedly calls code/run_experiments.py and parses the printed statistics into testData.csv
./testSuite.bash <path/to/instanceFolder/> <solver> <#problems> <timeout> <low_level_type> <merge_threshold>
- <solver>: Currently MA-CBS.
- <#problems>: Number of instances to run.
- <timeout>: per-instance time limit in seconds.
- <low_level_type>: joint for joint A* or nested for nested CBS.
- <merge_threshold>: MA-CBS merge threhsold B.
The script saves the raw console output to logs/log<i>.
After all runs finish, parses the logs and appends one CSV line per instance to testData.csv.

## Output
Each testData.csv row has the form:
<low_level_type>, <CPUtime>, <SUMcost>, <EXPHLnode>, <GENHLnode>,
<LLcalls>, <LLTEcall>, <LLAEcall>, <LLPOsize>, <TOTcost>, <instance_index>
This can be loaded into Excel, Google Sheets, and other platforms to reproduce the plots in the report.



