// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.7.0;

import "./Stages.sol";
import "./Microgrids.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract Members is Ownable, Stage, Microgrids {

    address[] private members_;

    // Used to avoid overflow while using uint256.

    using SafeMath for uint256;

    // A member belongs to a microgrid, signs to agree on the conditions, sets a buying price, sets a selling price.

    struct Member {
        uint256 microgridId_;
        bool signed_;
        uint256 buyPrice_;
        uint256 sellPrice_;
        uint256 soldEnergy_;
        uint256[] certificates_;
        bool exist_;
    }

    mapping(address => Member) private memberInfo_;

    event addressSigned(address signed);

    function registerMember(
        address _member, uint256 _microgridId, uint256 _buyPrice, uint256 _sellPrice
    ) public onlyOwner atStage(Stages.CREATION) microgridExists(_microgridId) {

        require(!memberInfo_[_member].exist_, "Member already exist.");

        members_.push(_member);
        memberInfo_[_member].microgridId_ = _microgridId;
        memberInfo_[_member].signed_ = false;
        memberInfo_[_member].buyPrice_ = _buyPrice;
        memberInfo_[_member].sellPrice_ = _sellPrice;
        memberInfo_[_member].soldEnergy_ = 0;
        memberInfo_[_member].exist_ = true;
    
    }

    function signMember(address _member) internal {

        // setStage(Stages.SIGN);

        require(memberInfo_[_member].exist_, "Invalid verification");
        require(!memberInfo_[_member].signed_, "Already signed");
        
        memberInfo_[_member].signed_ = true;
        
        emit addressSigned(_member);

    }

    function getMember(address _member) public view returns (uint256, uint256, bool) {

        Member memory p = memberInfo_[_member];
        return (p.buyPrice_, p.sellPrice_, p.exist_);
    
    }

    function getMembers() public view onlyOwner returns (address[] memory) {
    
        return members_;
    
    }

    function allSigned() internal view returns (bool) {

        bool start = false;

        for (uint256 i = 0; i < members_.length; i++) {
            start = start && memberInfo_[members_[i]].signed_;
        }

        return start;

    }

}