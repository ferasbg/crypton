import argparse
parser = argparse.ArgumentParser(description="Crypton")
parser.add_argument(
    "--num-clients",
    default=10,
    type=int,
)

args = parser.parse_args()
main(args)
