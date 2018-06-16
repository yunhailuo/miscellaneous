# Toggle Format JSON (toggle_format_json)
This is a chrome extension for getting JSON object of a ENCODE portal page.
  * This extension can be activated by clicking the icon in toolbar or by keyboard shortcut "Ctrl+Shift+Y".
  * This extension only works on ENCODE website (\*.encodeproject.org or \*.encodedcc.org).
  * If the page is a plain (rendered) page, this extension will try to open the corresponding JSON page by appending one parameter "format=json" to the url.
  * If the page is a JSON page, this extension will try to open the corresponding rendered page by removing "format=json" in the url.
  * The new page will be opened in the same tab.
