# Use Windows NCSI for automatic wifi authentication
### Problem description
&nbsp;&nbsp;&nbsp;&nbsp;My university implemented a captive portal on the wifi which requires an authentication with about 3-4 hours timeout. The authentication can be done through a simply https request to a specific webpage. In order to save myself from having to logon every 3-4 hours, I would ask the computer to automatically logon once the connection is lost. This thought comes from the observation that Windows 10 knows about it when I need authentication. It pops up a balloon telling me "Action required to use this network". It also opens a page which is then redirected by the captive portal to the university's authentication page. It looks like I'm almost there. What I need is just to replace the page Windows 10 opens with the https request which help me finish the authentication. It turns out, though, a little bit more than that.
### Solution and its related knowledge
1. Network Connectivity Status Indicator (NCSI)
The first question I have is how Windows 10 knows that I need authentication. It actually did that through the [Network Connectivity Status Indicator feature](https://technet.microsoft.com/en-us/library/ee126135(v=ws.10).aspx) which starts in Windows Vista. According to [nhinkle's research](http://blog.superuser.com/2011/05/16/windows-7-network-awareness/), NCSI first sends a request for http://www.msftncsi.com/ncsi.txt and http://ipv6.msftncsi.com/ncsi.txt. It expects a 200 OK response header with the proper text ("Microsoft NCSI" without quotes or new line or other non-printing characters) returned. If the response is never received, or if there is a redirect, then NCSI sends a request for DNS name resolution of dns.msftncsi.com. It expects that the resolution of the DNS name to: **131.107.255.255** or **fd3e:4f5a:5b81::1**. If DNS resolves properly but the page is inaccessible, then it is assumed that there is a working internet connection, but an in-browser authentication page is blocking access to the file. This results in the pop-up balloon. If DNS resolution fails or returns the wrong address, then it is assumed that the internet connection is completely unsuccessful, and the “no internet access” error is shown.
2. NCSI logs events in Event Viewer in Windows 10?
Obviously, the next question is how I can utilize this feature. According to [the official documentation](https://technet.microsoft.com/en-us/library/ee126135(v=ws.10).aspx), "NCSI does not log events in Event Viewer" "in Windows 7 and Windows Server 2008 R2". I couldn't find any sayings for Windows 10. But I did find in "Event Viewer" that there is a log for "Applications and Services Logs-Microsoft-Windows-NCSI/Operational". It is now possible to attach a task to the "require log on" event.
3. How to filter out the correct event for "require log on"?
It is easy to narrow down relevant events based the time when the balloon pops up. After looking into event details, the exact event has Event ID 4042 and CapabilityChangeReason 7 (ActiveHttpProbeFailedHotspotDetected). "Create Basic Task Wizard" only filter events based on Log and Event ID. This is not enough because events we don't want ("ActiveHttpProbeSucceeded", "CapabilityReset", etc) are also recorded under ID 4042.   
&nbsp;&nbsp;&nbsp;&nbsp;A Windows task is kept in XML format. However, you cannot edit the XML file (under C:\Windows\System32\Tasks) directly for security reason (maybe a hash check?). You can either import a task from a XML file or perform some XML editing in task scheduler. A [event triggered task](https://msdn.microsoft.com/en-us/library/windows/desktop/aa446889(v=vs.85).aspx) has an EventTrigger object. This object has a child element of event subscription query. [Event Selection](https://msdn.microsoft.com/en-us/library/aa385231(VS.85).aspx) will be performed here. "Filters can be used in XPath event queries." Therefore, we can filter out the correct event for "require log on" like this:
```
<QueryList>
  <Query Id="0" Path="Microsoft-Windows-NCSI/Operational">
    <Select Path="Microsoft-Windows-NCSI/Operational">
        *[System[(EventID=4042)] and EventData[Data[@Name="CapabilityChangeReason"]=7]]
    </Select>
  </Query>
</QueryList>
```
4. The action for http request.
Windows CMD.EXE/COMMAND.COM doesn't have commands for http request. PowerShell's Invoke-WebRequest (alias: iwr, curl, wget) will be used. "powershell" is a command that can be recognized by task action (without absolute path). For authentication, I need to submit a https request but don't really need the response for anything.
```
powershell -NoLogo -NonInteractive "& {Invoke-WebRequest https://auth.myuniversity.edu/logon > $NULL}"
```
&nbsp;&nbsp;&nbsp;&nbsp;Other settings for the task can use defaults
### A new problem and its temporary solution
**Problem**  
&nbsp;&nbsp;&nbsp;&nbsp;Now, the wifi balloon still pops up and the http request is triggered correctly. However, Windows now disconnects itself from the network even though I've already been authenticated. I will need to connect again manually. I don't need to log on though because the automatic task works which gets me authenticated. [Similar problems](https://superuser.com/questions/1169229/why-does-windows-connect-to-a-network-automatically-and-shortly-thereafter-disco) have been seen. A general conclusion is that this comes from the school's Captive Portal. I've also looked into things like "[disable IEEE 802.1X authentication](https://answers.microsoft.com/en-us/windows/forum/windows_7-networking/network-keeps-disconnecting-from-laptop-windows-7/1338d169-9fcc-4a05-87e8-60e0f1aa0bdf)". No luck there.  
**Temporary Solution**  
&nbsp;&nbsp;&nbsp;&nbsp;The temporary solution still relies on Windows task. No more explains since there is nothing new (to me). The Event Filter is:
```
<QueryList>
  <Query Id="0" Path="Microsoft-Windows-NetworkProfile/Operational">
    <Select Path="Microsoft-Windows-NetworkProfile/Operational">
      *[System[(EventID=10001)] and EventData[Data[@Name="Name"]="myuniversity"] and EventData[Data[@Name="Description"]="myuniversity"]]
    </Select>
  </Query>
</QueryList>
```
&nbsp;&nbsp;&nbsp;&nbsp;The action is:
```
netsh wlan connect name=myuniversity
```
