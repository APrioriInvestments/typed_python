/**
 * We use a singleton registry object
 * where we make available all possible
 * Components. This is useful for Webpack,
 * which only bundles explicitly used
 * Components during build time.
 */
import {AsyncDropdown, AsyncDropdownContent} from './components/AsyncDropdown';
import {Badge} from './components/Badge';
import {Button} from './components/Button';
import {ButtonGroup} from './components/ButtonGroup';
import {Card} from './components/Card';
import {CardTitle} from './components/CardTitle';
import {CircleLoader} from './components/CircleLoader';
import {Clickable} from './components/Clickable';
import {Code} from './components/Code';
import {CodeEditor} from './components/CodeEditor';
import {CollapsiblePanel} from './components/CollapsiblePanel';
import {Columns} from './components/Columns';
import {Container} from './components/Container';
import {ContextualDisplay} from './components/ContextualDisplay';
import {Dropdown} from './components/Dropdown';
import {Expands} from './components/Expands';
import {HeaderBar} from './components/HeaderBar';
import {KeyAction} from './components/KeyAction';
import {LoadContentsFromUrl} from './components/LoadContentsFromUrl';
import {LargePendingDownloadDisplay} from './components/LargePendingDownloadDisplay';
import {Main} from './components/Main';
import {Modal} from './components/Modal';
import {Octicon} from './components/Octicon';
import {Padding} from './components/Padding';
import {Popover} from './components/Popover';
import {ResizablePanel} from './components/ResizablePanel';
import {RootCell} from './components/RootCell';
import {Sequence} from './components/Sequence';
import {Scrollable} from './components/Scrollable';
import {SingleLineTextBox} from './components/SingleLineTextBox';
import {Span} from './components/Span';
import {Subscribed} from './components/Subscribed';
import {SubscribedSequence} from './components/SubscribedSequence';
import {Table} from './components/Table';
import {Tabs} from './components/Tabs';
import {Text} from './components/Text';
import {Traceback} from './components/Traceback';
import {_NavTab} from './components/_NavTab';
import {Grid} from './components/Grid';
import {Sheet} from './components/Sheet';
import {Plot} from './components/Plot';
import {_PlotUpdater} from './components/_PlotUpdater';
import {Timestamp} from './components/Timestamp';
import {SplitView} from './components/SplitView';

const ComponentRegistry = {
    AsyncDropdown,
    AsyncDropdownContent,
    Badge,
    Button,
    ButtonGroup,
    Card,
    CardTitle,
    CircleLoader,
    Clickable,
    Code,
    CodeEditor,
    CollapsiblePanel,
    Columns,
    Container,
    ContextualDisplay,
    Dropdown,
    Expands,
    HeaderBar,
    KeyAction,
    LoadContentsFromUrl,
    LargePendingDownloadDisplay,
    Main,
    Modal,
    Octicon,
    Padding,
    Popover,
    ResizablePanel,
    RootCell,
    Sequence,
    Scrollable,
    SingleLineTextBox,
    Span,
    Subscribed,
    SubscribedSequence,
    Table,
    Tabs,
    Text,
    Traceback,
    _NavTab,
    Grid,
    Sheet,
    Plot,
    _PlotUpdater,
    Timestamp,
    SplitView
};

export {ComponentRegistry, ComponentRegistry as default};
