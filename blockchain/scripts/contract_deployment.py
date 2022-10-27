import time
import random

from brownie import Contract, accounts, config, network, project

if __name__ == '__main__':

    p = project.load('./blockchain/', name='BCTE')
    p.load_config()

    network.connect('development')

    # Deploy the contract

    print(p)

    market_contract = p.Market.deploy({'from': accounts[0]})

    accounts[0].transfer(market_contract.address, '1 ether')

    try:

        while True:

            time.sleep(1)

            accounts_indexes = range(len(accounts))
            from_, to_ = random.sample(accounts_indexes, 2)

            accounts[from_].transfer(accounts[to_], '1 ether')

    except KeyboardInterrupt:

        print('Cancelled by user')