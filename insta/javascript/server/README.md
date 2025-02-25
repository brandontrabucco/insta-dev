Author: [Brandon Trabucco](https://btrabuc.co)
Date: 2025-02-01

# Server Endpoint For Scaling Web Agents.

This Node.js server provides an endpoint for running web navigation agents, 
and can manage multiple concurrent browsing sessions, up to 
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