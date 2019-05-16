/**
 * For any given view, collects and
 * reports which Cell components are
 * not used.
 */

const CellTrackerDef = function(){
    this.used = new Set();
    this.unused = new Set();
    this.isTracking = false;

    this.start = function(){
        this.isTracking = true;
        this.intervalID = setInterval(this.update, 200);
    }.bind(this);

    this.serialize = function(){
        let storedDict = {};
        let stored = window.localStorage.getItem('trackedCells');
        if(stored){
            storedDict = JSON.parse(stored);
        }
        storedDict.used = Array.from(this.used);
        storedDict.unused = Array.from(this.unused);
        storedDict.isTracking = this.isTracking;
        window.localStorage.setItem('trackedCells', JSON.stringify(storedDict));
    }.bind(this);

    this.update = function(){
        let self = this;
        Object.keys(availableComponents).forEach(compName => {
            let results = document.querySelectorAll(`[data-cell-type="${compName}"]`);
            if(results.length > 0){
                self.used.add(compName);
            }
        });
        let allSet = new Set(Object.keys(availableComponents));
        this.unused = allSet - this.used;
        this.serialize();
    }.bind(this);

    this.stop = function(){
        this.isTracking = false;
        if(this.intervalID){
            window.clearInterval(this.intervalID);
        }
        this.serialize();
    }.bind(this);

    this.clear = function(){
        let stored = window.localStorage.removeItem('trackedCells');
    }.bind(this);

    this.onPageLoad = function(){
        // If there is data in localStorage and isTracking
        // is set to true, then we re-start tracking on
        // page load.
        let stored = window.localStorage.getItem('trackedCells');
        if(stored){
            let storedData = JSON.parse(stored);
            if(storedData.isTracking){
                this.start();
            }
        }
    }.bind(this);
};

document.addEventListener('DOMContentLoaded', () => {
    window.CellTracker = new CellTrackerDef();
    window.CellTracker.onPageLoad();
});
