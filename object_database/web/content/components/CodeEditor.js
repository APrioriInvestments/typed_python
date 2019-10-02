/**
 * CodeEditor Cell Component
 */

import {Component} from './Component';
import {h} from 'maquette';

class CodeEditor extends Component {
    constructor(props, ...args){
        super(props, ...args);
        this.editor = null;
        // used to schedule regular server updates
        this.SERVER_UPDATE_DELAY_MS = 1;
        //this.editorStyle = 'width:100%;min-height:100%;margin:auto;border:1px solid lightgray;';

        this.setupEditor = this.setupEditor.bind(this);
        this.setupKeybindings = this.setupKeybindings.bind(this);
        this.changeHandler = this.changeHandler.bind(this);

        // Used to register and deregister
        // any global KeyListener instance
        this._onBlur = this._onBlur.bind(this);
        this._onFocus = this._onFocus.bind(this);
    }

    componentDidLoad() {
        this.setupEditor();

        if (this.editor === null) {
            console.log("editor component loaded but failed to setup editor");
        } else {
            console.log("setting up editor");
            this.editor.last_edit_millis = Date.now();

            this.editor.setTheme("ace/theme/textmate");
            this.editor.session.setMode("ace/mode/python");
            this.editor.setAutoScrollEditorIntoView(true);
            this.editor.session.setUseSoftTabs(true);
            this.editor.setValue(this.props.extraData.initialText);

            if (this.props.extraData.autocomplete) {
                this.editor.setOptions({enableBasicAutocompletion: true});
                this.editor.setOptions({enableLiveAutocompletion: true});
            }

            if (this.props.extraData.noScroll) {
                this.editor.setOption("maxLines", Infinity);
            }

            if (this.props.extraData.fontSize !== undefined) {
                this.editor.setOption("fontSize", this.props.extraData.fontSize);
            }

            if (this.props.extraData.minLines !== undefined) {
                this.editor.setOption("minLines", this.props.extraData.minLines);
            } else {
                this.editor.setOption("minLines", Infinity);
            }

            this.setupKeybindings();

            this.changeHandler();
        }
    }


    build(){
        return h('div',
            {
                class: "cell code-editor",
                id: this.props.id,
                "data-cell-id": this.props.id,
                "data-cell-type": "CodeEditor",
            },
                 [h('div', { id: "editor" + this.props.id, class: "code-editor-inner" }, [])
        ]);
    }

    setupEditor(){
        let editorId = "editor" + this.props.id;
        // TODO These are global var defined in page.html
        // we should do something about this.

        // here we bing and inset the editor into the div rendered by
        // this.render()
        this.editor = ace.edit(editorId);
        // TODO: deal with this global editor list
        aceEditors[editorId] = this.editor;
    }

    changeHandler() {
	var editorId = this.props.id;
	var editor = this.editor;
	var SERVER_UPDATE_DELAY_MS = this.SERVER_UPDATE_DELAY_MS;
        this.editor.on('focus', this._onFocus);
        this.editor.on('blur', this._onBlur);
        this.editor.session.on(
            "change",
            function(delta) {
                // WS
                let responseData = {
                    event: 'editor_change',
                    'target_cell': editorId,
                    data: delta
                };
                cellSocket.sendString(JSON.stringify(responseData));
                //record that we just edited
                editor.last_edit_millis = Date.now();

		//schedule a function to run in 'SERVER_UPDATE_DELAY_MS'ms
		//that will update the server, but only if the user has stopped typing.
		// TODO unclear if this is owrking properly
		window.setTimeout(function() {
		    if (Date.now() - editor.last_edit_millis >= SERVER_UPDATE_DELAY_MS) {
			//save our current state to the remote buffer
			editor.current_iteration += 1;
			editor.last_edit_millis = Date.now();
			editor.last_edit_sent_text = editor.getValue();
			// WS
			let responseData = {
			    event: 'editing',
			    'target_cell': editorId,
			    buffer: editor.getValue(),
			    selection: editor.selection.getRange(),
			    iteration: editor.current_iteration
			};
			cellSocket.sendString(JSON.stringify(responseData));
		    }
		}, SERVER_UPDATE_DELAY_MS + 2); //note the 2ms grace period
            }
        );
    }

    setupKeybindings() {
        this.props.extraData.keybindings.map((kb) => {
            this.editor.commands.addCommand(
                {
                    name: 'cmd' + kb,
                    bindKey: {win: 'Ctrl-' + kb,  mac: 'Command-' + kb},
                    readOnly: true,
                    exec: () => {
                        this.editor.current_iteration += 1;
                        this.editor.last_edit_millis = Date.now();
                        this.editor.last_edit_sent_text = this.editor.getValue();

                        // WS
                        let responseData = {
                            event: 'keybinding',
                            'target_cell': this.props.id,
                            'key': kb,
                            'buffer': this.editor.getValue(),
                            'selection': this.editor.selection.getRange(),
                            'iteration': this.editor.current_iteration
                        };
                        cellSocket.sendString(JSON.stringify(responseData));
                    }
                }
            );
        });
    }

    _onBlur(event){
        if(this.constructor.keyListener){
            this.constructor.keyListener.start();
        }
    }

    _onFocus(event){
        if(this.constructor.keyListener){
            this.constructor.keyListener.pause();
        }
    }
}

export {CodeEditor, CodeEditor as default};
