import 'maquette';
const h = maquette.h;
//import {langTools} from 'ace/ext/language_tools';
import {CellHandler} from './CellHandler';
import {CellSocket} from './CellSocket';
import {ComponentRegistry} from './ComponentRegistry';
import {KeyListener} from './components/util/KeyListener';
import {Component} from './components/Component';

/**
 * Globals
 **/
window.langTools = ace.require("ace/ext/language_tools");
window.aceEditors = {};
window.handsOnTables = {};

/**
 * Initial Render
 **/
const initialRender = function(){
    return h("div", {}, [
         h("div", {id: "page_root"}, [
             h("div.container-fluid", {}, [
                 h("div.card", {class: "mt-5"}, [
                     h("div.card-body", {}, ["Loading..."])
                 ])
             ])
         ]),
         h("div", {id: "holding_pen", style: "display:none"}, [])
     ]);
};

/**
 * Cell Socket and Handler
 **/
let projector = maquette.createProjector();
const cellSocket = new CellSocket();
const cellHandler = new CellHandler(h, projector, ComponentRegistry);
cellSocket.onPostscripts(cellHandler.handlePostscript);
cellSocket.onMessage(cellHandler.receive);
cellSocket.onClose(cellHandler.showConnectionClosed);
cellSocket.onError(err => {
    console.error("SOCKET ERROR: ", err);
});

/** For now, we bind the current socket and handler to the global window **/
window.cellSocket = cellSocket;
window.cellHandler = cellHandler;

/** Render top level component once DOM is ready **/
document.addEventListener('DOMContentLoaded', () => {
    projector.append(document.body, initialRender);
    cellSocket.connect();
    Component.keyListener = new KeyListener();
    window._keyListener = Component.keyListener;
    Component.keyListener.start(document, cellSocket);
    window._keyListener = Component.keyListener;
});

// TESTING; REMOVE
console.log('Main module loaded');
