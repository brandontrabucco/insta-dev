"use strict";
/*

Author: [Brandon Trabucco](https://btrabuc.co)
Date: 2025-02-01

# Server Endpoint For Scaling Web Agents.

This Node.js server provides an endpoint for running web navigation agents,
and can manage multiple concurrent browsing ACTIVE_SESSIONS, up to
64 with a single running server in our tests.

Start the server with the following command:

```bash
node insta/javascript/server/src/index.js 3000
```

## Start a new browsing session and receive a session ID.

POST `/start?width=$WIDTH&height=$HEIGHT`

- Query parameters: `width`, `height`
    - `width`: viewport width in pixels (default: 1920)
    - `height`: viewport height in pixels (default: 1080)

- JSON body: `browser_kwargs`, `context_kwargs`
    - `browser_kwargs`: dictionary of browser launch options
    - `context_kwargs`: dictionary of context options

- Return value: `session_id`
    - `session_id`: unique session ID for the browsing session

---

## Close the browsing session and release resources.

POST `/close?session_id=$SESSION_ID`:

- Query parameters: `session_id`
    - `session_id`: unique session ID for the browsing session

---

## Load a URL in the browsing session.

POST `/goto?url=$URL&session_id=$SESSION_ID`

- Query parameters: `session_id`, `url`
    - `session_id`: unique session ID for the browsing session
    - `url`: URL to load in the browsing session

---

## Extract metadata from the webpage.

POST `/observation?session_id=$SESSION_ID`:

- Query parameters: `session_id`
    - `session_id`: unique session ID for the browsing session

- Return value: `playwright_observation`
    - `playwright_observation`: dictionary containing the following keys:
        - `raw_html`: raw HTML content of the webpage
        - `screenshot`: base64-encoded PNG screenshot of the webpage
        - `metadata`: dictionary of metadata for each DOM node
        - `current_url`: current URL of the webpage (after redirects)

---

## Execute an action in the browsing session.

POST `/action?session_id=$SESSION_ID`:

- Query parameters: `session_id`
    - `session_id`: unique session ID for the browsing session

- JSON body: `action`
    - `action`: list of dictionaries containing the following keys for each function call:
        - `dotpath`: dot-separated path to the function in the Playwright API
        - `args`: string containing function arguments
 
 */
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const playwright_extra_1 = require("playwright-extra");
const puppeteer_extra_plugin_stealth_1 = __importDefault(require("puppeteer-extra-plugin-stealth"));
const crypto_1 = __importDefault(require("crypto"));
const vm_1 = __importDefault(require("vm"));
// register the Playwright Stealth plugin
playwright_extra_1.chromium.use((0, puppeteer_extra_plugin_stealth_1.default)());
// start the playwright server and track active browsing sessions
// each browsing session is assigned a unique session ID
const APP = (0, express_1.default)();
const ACTIVE_SESSIONS = {};
// Middleware to parse JSON bodies
// handles action post requests with JSON bodies
APP.use(express_1.default.json());
// configure the default server port and accept a custom port
// useful for spawning multiple Playwright servers
const DEFAULT_SERVER_PORT = 3000;
const PORT = process.argv[2] || DEFAULT_SERVER_PORT;
// each browsing session is assigned a unique session ID
// to track the session and manage resources
const SESSION_ID_LENGTH = 512;
const SESSION_TIMEOUT_THRESHOLD = 30 * 60 * 1000;
const SESSION_TIMEOUT_INTERVAL = 30 * 1000;
// configure the default viewport size for the browsing session
// standard desktop browsing resolution
const DEFAULT_WIDTH = 1920;
const DEFAULT_HEIGHT = 1080;
// restrict the maximum number of nodes and HTML size
// and the maximum chained function calls
const MAX_NODE_SIZE = 10000;
const MAX_HTML_SIZE = 10000000;
const MAX_CHAINED_CALLS = 3;
// skip certain tags when extracting metadata
// these tags contain little useful information for agents
const SKIP_TAGS = [
    'HTML',
    'HEAD',
    'TITLE',
    'BODY',
    'SCRIPT',
    'STYLE',
    'LINK',
    'META',
    'NOSCRIPT',
    'IFRAME',
    'FRAME',
    'FRAMESET'
];
// generate a random session ID for tracking the browsing session
// agents will send their session ID when querying their browsing session
const generate_session_id = () => {
    const session_id_bytes = crypto_1.default.randomBytes(Math.ceil(SESSION_ID_LENGTH / 4 * 3));
    const session_id = (session_id_bytes.toString('base64')
        .replace(/\+/g, 'A').replace(/\//g, 'A').replace(/=+$/, 'A')
        .slice(0, SESSION_ID_LENGTH));
    return session_id;
};
// extract metadata from the webpage for agents
// includes all data needed to reconstruct the webpage
const process_observation = ([MAX_NODE_SIZE, MAX_HTML_SIZE, SKIP_TAGS]) => {
    function elementFromPoint(x, y) {
        var _a, _b;
        let node = document.elementFromPoint(x, y);
        let child = (_a = node === null || node === void 0 ? void 0 : node.shadowRoot) === null || _a === void 0 ? void 0 : _a.elementFromPoint(x, y);
        while (child && child !== node) {
            node = child;
            child = (_b = node === null || node === void 0 ? void 0 : node.shadowRoot) === null || _b === void 0 ? void 0 : _b.elementFromPoint(x, y);
        }
        return child || node;
    }
    const metadata = {};
    const preprocess_node = (node, backend_node_id) => {
        if (node.tagName in SKIP_TAGS) {
            return;
        }
        node.setAttribute('backend_node_id', backend_node_id.toString());
        const bounding_client_rect = node.getBoundingClientRect();
        const raw_computed_style = window.getComputedStyle(node);
        const computed_style = {};
        for (let idx = 0; idx < raw_computed_style.length; idx++) {
            const propertyName = raw_computed_style[idx];
            computed_style[propertyName] = (raw_computed_style
                .getPropertyValue("" + propertyName));
        }
        const scroll_left = node.scrollLeft;
        const scroll_top = node.scrollTop;
        let editable_value = null;
        if (node.tagName === 'INPUT') {
            editable_value = node.value;
        }
        else if (node.tagName === 'TEXTAREA') {
            editable_value = node.value;
        }
        else if (node.tagName === 'SELECT') {
            editable_value = node.selectedIndex;
        }
        else if (node.getAttribute('contenteditable') === 'true') {
            editable_value = node.innerText;
        }
        const is_visible = node.checkVisibility({
            contentVisibilityAuto: true,
            opacityProperty: true,
            visibilityProperty: true,
        });
        let is_frontmost = false;
        let top_elem_outer_html = '';
        if (is_visible) {
            let top_element = elementFromPoint(bounding_client_rect.x +
                bounding_client_rect.width / 2, bounding_client_rect.y +
                bounding_client_rect.height / 2);
            if (top_element) {
                while (top_element) {
                    const top_element_is_visible = top_element.checkVisibility({
                        contentVisibilityAuto: true,
                        opacityProperty: true,
                        visibilityProperty: true,
                    });
                    if (top_element_is_visible ||
                        top_element.parentElement === null ||
                        top_element.parentElement === document.body) {
                        break;
                    }
                    top_element = (top_element.parentElement);
                }
                is_frontmost = (node === top_element ||
                    node.contains(top_element) ||
                    top_element.contains(node));
                top_elem_outer_html = (top_element.outerHTML);
            }
        }
        metadata[backend_node_id] = {
            'backend_node_id': backend_node_id,
            'bounding_client_rect': bounding_client_rect,
            'computed_style': computed_style,
            'scroll_left': scroll_left,
            'scroll_top': scroll_top,
            'editable_value': editable_value,
            'is_visible': is_visible,
            'is_frontmost': is_frontmost,
            'top_elem_outer_html': top_elem_outer_html
        };
    };
    let allNodes = Array.from(document.body.getElementsByTagName('*'));
    allNodes = allNodes.slice(0, MAX_NODE_SIZE);
    allNodes.forEach(preprocess_node);
    let raw_html = document.documentElement.outerHTML;
    raw_html = raw_html.slice(0, MAX_HTML_SIZE);
    return [metadata, raw_html];
};
// resolve the dotpath to a function in the Playwright API
// allows agents to call arbitrary functions within the Playwright API
const resolve_dotpath = (target_module, dotpath_parts) => {
    for (let path_idx = 0; path_idx < dotpath_parts.length; path_idx++) {
        const dotpath_part = dotpath_parts[path_idx];
        if (target_module === undefined || target_module === null) {
            throw new Error('Target module is not defined');
        }
        else if (dotpath_part === undefined || dotpath_part === null) {
            throw new Error('Dotpath part is not defined');
        }
        else if (typeof dotpath_part !== 'string') {
            throw new Error('Dotpath part is not a string');
        }
        else if (target_module[dotpath_part] === undefined || target_module[dotpath_part] === null) {
            throw new Error('Dotpath part does not exist: ' + dotpath_part);
        }
        if (typeof target_module[dotpath_part] === 'function') {
            target_module = (target_module[dotpath_part]
                .bind(target_module));
        }
        else {
            target_module = target_module[dotpath_part];
        }
        if (target_module === undefined || target_module === null) {
            throw new Error('Target module is currently null: ' +
                dotpath_parts.join('.'));
        }
    }
    return target_module;
};
// parse function arguments in a safe context using the JavaScript VM
// prevents access to sensitive Node.js APIs
function safe_parse_args(args) {
    "use strict";
    return vm_1.default.runInNewContext(`[${args}]`, Object.create(null));
}
// adjust arguments and handle edge cases for certain functions
// for example, remapping relative URLs to absolute URLs
// TODO: we can add more cases here in the future
const prepare_args = (target_module, dotpath_parts, args, browser, context, page) => {
    // remap relative URLs to absolute URLs based on the current URL
    // for the goto function
    if (dotpath_parts[dotpath_parts.length - 1] === 'goto' && args.length > 0) {
        const current_url = page.url();
        const new_url = args[0];
        args[0] = new URL(new_url, current_url).toString();
    }
    return args;
};
// perform an action in the browsing session by making function calls
// for each call, lookup the corresponding function in the Playwright API using its dotpath,
// and safely parse arbitrary string arguments from the agent
const playwright_function_call = (action, browser, context, page) => __awaiter(void 0, void 0, void 0, function* () {
    let target_module = {
        browser: browser,
        context: context,
        page: page
    };
    for (let action_idx = 0; action_idx < action.length; action_idx++) {
        // locate the target module within the Playwright API
        // and make a function call
        const dotpath_parts = (action[action_idx]['dotpath']
            .split('.'));
        target_module = resolve_dotpath(target_module, dotpath_parts);
        const args = prepare_args(target_module, dotpath_parts, safe_parse_args(action[action_idx]['args']), browser, context, page);
        target_module = yield target_module(...args);
    }
    return target_module;
});
// start a new browsing session for a swarm of agents
// agents will make a post request to this endpoint and receive a session ID
APP.post('/start', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    let width, height;
    try {
        width = parseInt(req.query.width ||
            DEFAULT_WIDTH.toString());
        height = parseInt(req.query.height ||
            DEFAULT_HEIGHT.toString());
        if (width <= 0 || height <= 0) {
            throw new Error('Invalid height or width');
        }
    }
    catch (error) {
        res.status(400).send('Invalid height or width');
        return;
    }
    let browser_kwargs = {};
    let context_kwargs = {};
    // read the action json from the request body
    if (req.body !== undefined && req.body !== null) {
        if ('browser_kwargs' in req.body) {
            browser_kwargs = req.body['browser_kwargs'];
        }
        if ('context_kwargs' in req.body) {
            context_kwargs = req.body['context_kwargs'];
        }
    }
    let browser;
    let context;
    let page;
    try {
        browser = yield playwright_extra_1.chromium.launch(Object.assign({ headless: true }, browser_kwargs));
        context = yield browser.newContext(Object.assign({}, context_kwargs));
        page = yield browser.newPage();
    }
    catch (error) {
        res.status(400).send('Failed to start browser: ' + error);
        return;
    }
    if (browser === undefined || page === undefined) {
        res.status(400).send('Failed to start browser');
        return;
    }
    try {
        yield page.setViewportSize({
            width: width, height: height
        });
    }
    catch (error) {
        res.status(400).send('Failed to set viewport size: ' + error);
        return;
    }
    const session_id = generate_session_id();
    console.log("Starting new session: " +
        session_id);
    const session_data = {
        browser: browser,
        context: context,
        page: page,
        timestamp: Date.now()
    };
    ACTIVE_SESSIONS[session_id] = session_data;
    res.status(200).send(session_id);
}));
// close the browsing session and release the resources
// resources associated with this session ID will be released
APP.post('/close', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const session_id = req.query.session_id;
    if (session_id === undefined) {
        res.status(400).send('Session ID not provided');
        return;
    }
    const session_data = ACTIVE_SESSIONS[session_id];
    if (session_data === undefined) {
        res.status(400).send('Session ID not found');
        return;
    }
    const browser = session_data.browser;
    const context = session_data.context;
    const page = session_data.page;
    if (browser === undefined || context === undefined || page === undefined) {
        res.status(400).send('Session ID not found');
        return;
    }
    try {
        yield page.close();
        yield context.close();
        yield browser.close();
        delete ACTIVE_SESSIONS[session_id];
    }
    catch (error) {
        res.status(400).send('Failed to close session: ' + error);
        return;
    }
    res.status(200).send('Session successfully closed');
}));
// load a URL in the queried browsing session
// agents can subsequently post to /observation to extract metadata
APP.post('/goto', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const session_id = req.query.session_id;
    const url = req.query.url;
    if (session_id === undefined) {
        res.status(400).send('Session ID not provided');
        return;
    }
    if (url === undefined) {
        res.status(400).send('URL not provided');
        return;
    }
    const session_data = ACTIVE_SESSIONS[session_id];
    if (session_data === undefined) {
        res.status(400).send('Session ID not found');
        return;
    }
    const browser = session_data.browser;
    const context = session_data.context;
    const page = session_data.page;
    session_data.timestamp = Date.now();
    if (browser === undefined || context === undefined || page === undefined) {
        res.status(400).send('Session ID not found');
        return;
    }
    try {
        yield page.goto(url);
    }
    catch (error) {
        res.status(400).send('Failed to load URL: ' + error);
        return;
    }
    res.status(200).send('Page successfully loaded');
}));
// preprocess the webpage and extract metadata needed for agents
// includes all data needed to reconstruct the webpage
APP.post('/observation', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const session_id = req.query.session_id;
    if (session_id === undefined) {
        res.status(400).send('Session ID not provided');
        return;
    }
    const session_data = ACTIVE_SESSIONS[session_id];
    if (session_data === undefined) {
        res.status(400).send('Session ID not found');
        return;
    }
    const browser = session_data.browser;
    const context = session_data.context;
    const page = session_data.page;
    session_data.timestamp = Date.now();
    if (browser === undefined || context === undefined || page === undefined) {
        res.status(400).send('Session ID not found');
        return;
    }
    let metadata;
    let raw_html;
    try {
        [metadata, raw_html] = yield page.evaluate(process_observation, [
            MAX_NODE_SIZE, MAX_HTML_SIZE, SKIP_TAGS
        ]);
    }
    catch (error) {
        res.status(400).send('Failed to extract metadata: ' + error);
        return;
    }
    if (metadata === undefined || metadata === null || raw_html === undefined || raw_html === null) {
        res.status(400).send('Failed to extract metadata and HTML');
        return;
    }
    let screenshot_bytes;
    try {
        screenshot_bytes = yield page.screenshot({
            type: 'png'
        });
    }
    catch (error) {
        res.status(400).send('Failed to capture screenshot: ' + error);
        return;
    }
    if (screenshot_bytes === undefined) {
        res.status(400).send('Failed to capture screenshot');
        return;
    }
    const screenshot_base64 = screenshot_bytes.toString('base64');
    let current_url;
    try {
        current_url = page.url();
    }
    catch (error) {
        res.status(400).send('Failed to extract current URL: ' + error);
        return;
    }
    if (current_url === undefined) {
        res.status(400).send('Failed to extract current URL');
        return;
    }
    const playwright_observation = {
        'raw_html': raw_html,
        'screenshot': screenshot_base64,
        'metadata': metadata,
        'current_url': current_url
    };
    try {
        res.status(200).send(playwright_observation);
    }
    catch (error) {
        res.status(400).send('Failed to send observation: ' + error);
    }
}));
// execute an action in the browsing session by making function calls
// agents must post a json object in the format: [{ "dotpath": "page.locator", "args": "[backend_node_id='5']" }]
//   - `dotpath`: a string representing the path to the function in the Playwright API
//   - `args`: a string representing arguments to pass to the function
APP.post('/action', (req, res) => __awaiter(void 0, void 0, void 0, function* () {
    const session_id = req.query.session_id;
    if (session_id === undefined) {
        res.status(400).send('Session ID not provided');
        return;
    }
    const session_data = ACTIVE_SESSIONS[session_id];
    if (session_data === undefined) {
        res.status(400).send('Session ID not found');
        return;
    }
    const browser = session_data.browser;
    const context = session_data.context;
    const page = session_data.page;
    session_data.timestamp = Date.now();
    if (browser === undefined || context === undefined || page === undefined) {
        res.status(400).send('Session ID not found');
        return;
    }
    // read the action json from the request body
    const action = req.body;
    if (action === undefined) {
        res.status(400).send('Action not provided');
        return;
    }
    // check that actions is a list dotpaths and args
    if (!Array.isArray(action)) {
        res.status(400).send('Action must be a list of dotpaths and args');
        return;
    }
    if (action.length === 0) {
        res.status(400).send('Action must contain at least one function call');
        return;
    }
    if (action.length > MAX_CHAINED_CALLS) {
        res.status(400).send('Action must contain at most 3 function calls');
        return;
    }
    for (let idx = 0; idx < action.length; idx++) {
        const function_call = action[idx];
        // check that each action item is an object with a dotpath and args
        if (typeof function_call !== 'object') {
            res.status(400).send('Action item must be an object');
            return;
        }
        if (function_call['dotpath'] === undefined) {
            res.status(400).send('Action item must contain a dotpath');
            return;
        }
        if (function_call['args'] === undefined) {
            res.status(400).send('Action item must contain args');
            return;
        }
        // check that the dotpath is a string
        if (typeof function_call['dotpath'] !== 'string') {
            res.status(400).send('Dotpath must be a string');
            return;
        }
        // check that the args is a string
        if (typeof function_call['args'] !== 'string') {
            res.status(400).send('Args must be a string');
            return;
        }
    }
    try {
        // Execute the provided javascript in a separate namespace
        // with the browser, context, and page objects
        const result = yield playwright_function_call(action, browser, context, page);
    }
    catch (error) {
        res.status(400).send('Failed to execute action: ' + error);
        return;
    }
    res.status(200).send('Action successfully executed');
}));
// start the Playwright server and listen on the specified port
// currently only accepts POST requests
APP.listen(PORT, () => __awaiter(void 0, void 0, void 0, function* () {
    return console.log(`Serving Playwright: http://localhost:${PORT}`);
}));
// check for idle sessions and close them after a timeout
// prevents memory leaks and resource exhaustion
setInterval(() => __awaiter(void 0, void 0, void 0, function* () {
    const sessions_to_remove = [];
    for (const session_id in ACTIVE_SESSIONS) {
        const session_data = ACTIVE_SESSIONS[session_id];
        if (session_data === undefined) {
            continue;
        }
        const session_expired = ((Date.now() - session_data.timestamp) >
            SESSION_TIMEOUT_THRESHOLD);
        if (session_expired) {
            sessions_to_remove.push(session_id);
        }
    }
    for (let idx = 0; idx < sessions_to_remove.length; idx++) {
        const session_data = ACTIVE_SESSIONS[sessions_to_remove[idx]];
        if (session_data === undefined) {
            continue;
        }
        const browser = session_data.browser;
        const context = session_data.context;
        const page = session_data.page;
        if (browser === undefined || context === undefined || page === undefined) {
            continue;
        }
        yield page.close();
        yield context.close();
        yield browser.close();
        delete ACTIVE_SESSIONS[sessions_to_remove[idx]];
        console.log("Closing idle session: " +
            sessions_to_remove[idx]);
    }
}), SESSION_TIMEOUT_INTERVAL);
