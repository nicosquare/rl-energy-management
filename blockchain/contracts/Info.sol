// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.7.0;

contract Info {

    struct PayInfo {
        uint256 total_;
        address investor_;
    }

    PayInfo private info;

    function setTotal(address _investor) internal {

        info.total_ = address(this).balance;
        info.investor_ = _investor;

    }

    function getInvestor() public view returns (address) {

        return info.investor_;

    }

    function getTotal() internal view returns (uint256) {

        return info.total_;

    }

    modifier withFunds() {

        require(info.total_ > 0, "Contract balance is 0");
        _;

    }
}