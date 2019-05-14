/**
 * Generic base Cell Component.
 * Should be extended by other
 * Cell classes on JS side.
 */

// NOTE: For the moment we assume global
// availability of the `h` hyperscript
// constructor.

class Component {
    constructor(props = {}, children = []){
        this.props = props;

        // Ensure that when we attempt to get
        // the `contentsId` prop, we append the
        // hacky bit of string to the plain id,
        // if present
        Object.defineProperty(this.props, 'contentsId', {
            get: function(){
                if(this.id){
                    return `${this.id}_____contents__`;
                }
                return undefined;
            }
        });
        this.children = children;
    }

    render(){
        // Objects that extend from
        // me should override this
        // method in order to generate
        // some content for the vdom
        throw new Error('You must implement a `render` method on Component objects!');
    }
}
