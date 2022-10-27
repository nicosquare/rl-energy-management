// SPDX-License-Identifier: GPL-3.0

pragma solidity >=0.7.0;

import "./Stages.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/math/SafeMath.sol";

contract Microgrids is Ownable, Stage{

    // Used to avoid overflow while using uint256.

    using SafeMath for uint256;

    // Single number id for each microgrid

    uint[] private microgrids_;

    // Microgid has a name, a latitude and a longitude

    struct Microgrid {
        string name_;
        string latitude_;
        string longitude_;
        bool exist_;
    }

    mapping(uint256 => Microgrid) private microgridInfo_;

    function registerMicrogrid(
        uint256 _microgrid, string memory _name, string memory _latitude, string memory _longitude
    ) public onlyOwner atStage(Stages.CREATION) {

        require(!microgridInfo_[_microgrid].exist_, "Microgrid already exist.");

        microgrids_.push(_microgrid);
        microgridInfo_[_microgrid].name_ = _name;
        microgridInfo_[_microgrid].latitude_ = _latitude;
        microgridInfo_[_microgrid].longitude_ = _longitude;
        microgridInfo_[_microgrid].exist_ = true;
    
    }

    function getMicrogrid(uint256 _microgrid) public view returns (string memory, string memory, string memory, bool) {

        Microgrid memory p = microgridInfo_[_microgrid];
        return (p.name_, p.latitude_, p.longitude_, p.exist_);

    }

    function getMicrogrids() public view onlyOwner returns (uint256[] memory) {

        return microgrids_;
    
    }

    modifier microgridExists(uint256 _microgrid) {
        require(microgridInfo_[_microgrid].exist_, "Microgrid does not exist.");
        _;
    }

}