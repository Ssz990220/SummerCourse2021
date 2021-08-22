from runner import Runner
import argparse


args = argparse.Namespace(scenario="cliffwalking",algo='sarsa',reload_config=False)


print("================== args: ", args)
print("== args.reload_config: ", args.reload_config)

runner = Runner(args)
runner.evaluate(10)