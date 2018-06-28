"use strict";

// Restrain extension action to certain host matched sites
function restrain_action() {
  chrome.storage.sync.get("hostSuffix", function(result) {
    let hostMatch = [], matcher;
    result.hostSuffix.forEach(function(item) {
      matcher = new chrome.declarativeContent.PageStateMatcher({
        pageUrl: {hostSuffix: item},
      });
      hostMatch.push(matcher);
    });
    chrome.declarativeContent.onPageChanged.removeRules(undefined, function() {
      chrome.declarativeContent.onPageChanged.addRules([{
        conditions: hostMatch,
        actions: [new chrome.declarativeContent.ShowPageAction()]
      }]);
    });
  });
}

chrome.runtime.onInstalled.addListener(function(details) {
  if (details.reason === "install") {
    chrome.storage.sync.set({
      params: {
        format: "json"
      },
      hostSuffix: [
        ".encodeproject.org",
        ".encodedcc.org"
      ]
    });
    restrain_action();
  }
});

chrome.pageAction.onClicked.addListener(function(activeTab){
  chrome.storage.sync.get("params", function(result) {
    let params = result.params;
    let url = new URL(activeTab.url);
    for (let key in params) {
      if (url.searchParams.get(key)) {
        url.searchParams.delete(key);
      } else {
        url.searchParams.set(key, params[key]);
      }
    }
    chrome.tabs.update(activeTab.id, { url: url.href });
  });
});
