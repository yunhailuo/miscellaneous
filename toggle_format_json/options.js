"use strict"

// Restores "hostsuffix" textarea using the preferences stored in
// chrome.storage.
function restore_options() {
  chrome.storage.sync.get("hostSuffix", function(result) {
    result.hostSuffix.forEach(function(item) {
      document.getElementById('hostsuffix').value += item + '\n';
    });
  });
  chrome.storage.sync.get("params", function(result) {
    let param;
    for (let key in result.params) {
      param = key + '=' + result.params[key] + '\n';
      document.getElementById('urlparams').value += param;
    }
  });
}
document.addEventListener('DOMContentLoaded', restore_options);

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

// Saves host suffix for matching actionable sites
function save_hostsuffix() {
  let temp = document.getElementById('hostsuffix').value.split('\n');
  let hosts = []
  for(let h of temp)
    h.trim() && hosts.push(h.trim());
  chrome.storage.sync.set({ hostSuffix: hosts });
}

// Saves URL parameters to be toggled
function save_urlparams() {
  let temp = document.getElementById('urlparams').value.split(/&|\n/);
  let params = {}, k, v;
  for (let p of temp) {
    if( p.trim()) {
      [k, v] = p.split('=');
      params[k] = v;
    }
  }
  chrome.storage.sync.set({params: params});
}

// Saves options to chrome.storage
function save_options() {
  save_hostsuffix();
  restrain_action();
  save_urlparams();
}
document.getElementById('save').addEventListener('click', save_options);
