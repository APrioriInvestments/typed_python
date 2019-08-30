/**
 * PageView Cell Component
 * Used for dividing up main views,
 * with optional header and footer.
 */
import {Component} from './Component';
import {h} from 'maquette';

/**
 * About Replacements
 * ------------------
 * This component has three regular
 * replacements:
 * * `header`
 * * `main`
 * * `footer`
 */

/**
 * About Named Children
 * `header` - An optional header cell
 * `main` - A required main content cell
 * `footer` - An optional footer cell
 */
class PageView extends Component {
    constructor(props, ...args){
        super(props, ...args);

        // Bind component methods
        this.makeHeader = this.makeHeader.bind(this);
        this.makeMain = this.makeMain.bind(this);
        this.makeFooter = this.makeFooter.bind(this);
    }

    build(){
        return h('div', {
            id: this.props.id,
            'data-cell-id': this.props.id,
            'data-cell-type': "PageView",
            class: 'cell page-view'
        }, [
            this.makeHeader(),
            this.makeMain(),
            this.makeFooter()
        ]);
    }

    makeHeader(){
        let headerContent = null;
        if(this.usesReplacements){
            headerContent = this.getReplacementElementFor('header');
        } else {
            headerContent = this.renderChildNamed('header');
        }
        if(headerContent){
            return h('header', {
                class: 'page-view-header'
            }, [headerContent]);
        } else {
            return null;
        }
    }

    makeMain(){
        let mainContent;
        if(this.usesReplacements){
            mainContent = this.getReplacementElementFor('main');
        } else {
            mainContent = this.renderChildNamed('main');
        }
        return mainContent;
    }

    makeFooter(){
        let footerContent = null;
        if(this.usesReplacements){
            footerContent = this.getReplacementElementFor('footer');
        } else {
            footerContent = this.renderChildNamed('footer');
        }
        if(footerContent){
            return h('footer', {
                class: 'page-view-footer'
            }, [footerContent]);
        } else {
            return null;
        }
    }
}

export {PageView, PageView as default};
