'use strict';

chrome.runtime.onInstalled.addListener(function() {
  chrome.declarativeContent.onPageChanged.removeRules(undefined, function() {
    chrome.declarativeContent.onPageChanged.addRules([{
      conditions: [
        new chrome.declarativeContent.PageStateMatcher({
          pageUrl: {hostSuffix: '.encodeproject.org'},
        }),
        new chrome.declarativeContent.PageStateMatcher({
          pageUrl: {hostSuffix: '.encodedcc.org'},
        })
      ],
          actions: [new chrome.declarativeContent.ShowPageAction()]
    }]);
  });
});

chrome.pageAction.onClicked.addListener(function(activeTab){
  var oldURL = activeTab.url;
  var formatJSON = oldURL.indexOf('format=json');
  var param;
  if (formatJSON == -1)
  {
    if (oldURL.indexOf('?') > -1)
    {
      param = "&format=json";
    } else {
      param = "?format=json";
    }
    chrome.tabs.update(activeTab.id, { url: oldURL + param });
  } else if (formatJSON > 0) {
    if (oldURL.charAt(formatJSON - 1) == "&")
    {
      chrome.tabs.update(activeTab.id, { url: oldURL.replace("&format=json", "") });
    } else if (oldURL.charAt(formatJSON - 1) == "?") {
      if (oldURL.charAt(formatJSON + 11) == "&"){
        chrome.tabs.update(activeTab.id, { url: oldURL.replace("format=json&", "") });
      } else {
        chrome.tabs.update(activeTab.id, { url: oldURL.replace("?format=json", "") });
      }
    }
  }
});
