import csv
import textwrap

def generate_configs(csv_configs):
    constants = """
    """
    constants = textwrap.dedent(constants)

    with open(csv_configs, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader, start=1):
            with open(f'configs/config{i}.py', 'w') as f:
                f.write(constants)
                for key, value in row.items():
                    if value.replace('.', '', 1).replace("e-", '', 1).replace("e+", '', 1).isdigit():
                        f.write(f"{key} = {value}\n")
                    elif value.lower() == "true":
                        f.write(f"{key} = True\n")
                    elif value.lower() == "false":
                        f.write(f"{key} = False\n")
                    elif value.lower() == "none":
                        f.write(f"{key} = None\n")
                    else:
                        f.write(f'{key} = "{value}"\n')

                    
if __name__ == "__main__":
    generate_configs('experiments.csv')