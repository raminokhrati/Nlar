import os
import sys
from experiments.config.config import shared_path
os.chdir(shared_path)
import subprocess

# Define the list of experiments and their corresponding script paths
experiments = {
    "mnist_logistic": "experiments/classifications/mnist_logistic.py",
    "mnist_mlp2h": "experiments/classifications/mnist_mlp2h.py",
    "cifar10_mlp7h": "experiments/classifications/cifar10_mlp7h.py",
    "cifar1_vgg11": "experiments/classifications/cifar10_vgg11.py",
    "cartpole": "experiments/reinforcement-learning-ddqn/cartpole.py",
    "sensitivity_mlp2h": "experiments/sensitivity/sensitivity_mlp2h.py",
    "sensitivity_mlp7h": "experiments/sensitivity/sensitivity_mlp7h.py"
}

def run_experiment(exp_name):
    if exp_name in experiments:
        script_path = experiments[exp_name]
        print(f"Running experiment: {exp_name}")
        subprocess.run([sys.executable, script_path], check=True)
    else:
        print(f"Experiment {exp_name} not found!")


def run_all_experiments():
    for exp_name, script_path in experiments.items():
        run_experiment(exp_name)


def main():
    print("Available experiments:")
    for i, exp_name in enumerate(experiments, 1):
        print(f"{i}. {exp_name}")

    choice = input("\nEnter the experiment number to run (or 'all' to run all experiments): ").strip()

    if choice.lower() == "all":
        run_all_experiments()
    elif choice.isdigit() and int(choice) in range(1, len(experiments) + 1):
        selected_experiment = list(experiments.keys())[int(choice) - 1]
        run_experiment(selected_experiment)
    else:
        print("Invalid input. Please try again.")


if __name__ == "__main__":
    main()

# import subprocess
# import os
#
# # List of experiments and their paths relative to the project root
# experiments = {
#     "mnist_logistic": "experiments/classifications/mnist_logistic.py",
#     "cifar10_mlp7h": "experiments/classifications/cifar10_mlp7h.py",
#     "cartpole": "experiments/reinforcement-learning-ddqn/cartpole.py",
#     "sensitivity_mlp7h": "experiments/sensitivity/sensitivity_mlp7h.py"
# }
#
#
# def run_experiment(exp_name):
#     if exp_name in experiments:
#         script_path = experiments[exp_name]
#         project_root = os.path.dirname(os.path.abspath(__file__))  # Get current path of run_experiments.py
#         project_root = os.path.abspath(os.path.join(project_root, ".."))  # Move one level up to the project root
#
#         print(f"Running experiment: {exp_name}")
#
#         # Add the project root to PYTHONPATH
#         env = os.environ.copy()
#         env["PYTHONPATH"] = project_root
#
#         # Run the experiment with the correct PYTHONPATH
#         subprocess.run(["python", script_path], cwd=project_root, env=env, check=True)
#     else:
#         print(f"Experiment {exp_name} not found!")
#
#
# def run_all_experiments():
#     for exp_name in experiments:
#         run_experiment(exp_name)
#
#
# if __name__ == "__main__":
#     choice = input("Enter the experiment name to run (or 'all' to run all): ").strip()
#
#     if choice == "all":
#         run_all_experiments()
#     elif choice in experiments:
#         run_experiment(choice)
#     else:
#         print("Invalid choice, please enter a valid experiment name.")
