
== for the NUS-WIDE multi-label classification

1. Download the dataset into data/raw/
2. Run `./_gen.sh` to generate the lists and tf records
3. `../../compile_data.sh`
4. `./setup.sh` to setup the experiments
5. `../../trains.sh data/exps gpu-id` to run the experiments
