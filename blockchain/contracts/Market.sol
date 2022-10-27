// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.7.0;

import "./Members.sol";
import "./Microgrids.sol";
import "./Info.sol";
import "@openzeppelin/contracts/utils/escrow/Escrow.sol";

contract Market is Microgrids, Members, Info {

    Escrow private _escrow;

    constructor() {
        _escrow = new Escrow();
    }

    receive() external payable {
        setTotal(msg.sender);
    }

    function energyTransaction(address _to, uint256 _energyKWh, uint256 _price) public {

        uint256 _amount = _energyKWh * _price;
        _escrow.deposit{value: _amount}(_to);
        _escrow.withdraw(payable(_to));
   
    }

}