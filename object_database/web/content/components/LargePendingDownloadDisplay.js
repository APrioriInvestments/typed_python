/**
 * LargePendingDownloadDisplay Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

class LargePendingDownloadDisplay extends Component {
    constructor(props, ...args){
        super(props, ...args);
    }

    build(){
        return (
            h('div', {
                id: 'object_database_large_pending_download_text',
                "data-cell-id": this.props.id,
                "data-cell-type": "LargePendingDownloadDisplay",
                class: "cell"
            })
        );
    }
}

export {LargePendingDownloadDisplay, LargePendingDownloadDisplay as default};
