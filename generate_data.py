"""
generate_data.py — Export the Nevo (2000) fake-cereal data bundled with
pyblp to data/*.csv. Only needed to regenerate the CSVs; the repository
already includes them.

Run:  python generate_data.py       (from the repo root)
      pip install pyblp              (if not yet installed)
"""

import os

import pandas as pd
import pyblp

os.chdir(os.path.dirname(os.path.abspath(__file__)))   # ensure relative paths work
os.makedirs('data', exist_ok=True)

# Product data (shares, prices, characteristics, 20 demand instruments)
product_data = pd.read_csv(pyblp.data.NEVO_PRODUCTS_LOCATION)
product_data.to_csv('data/nevo_products.csv', index=False)
print('data/nevo_products.csv', product_data.shape)

# Agent data (demographics + simulation draws nodes0-3, 20 per market)
agent_data = pd.read_csv(pyblp.data.NEVO_AGENTS_LOCATION)
agent_data.to_csv('data/nevo_agents.csv', index=False)
print('data/nevo_agents.csv', agent_data.shape)
