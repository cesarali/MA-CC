# Restricted HTTP CONNECT bridge from WSL localhost to the Windows VPN stack.
# This process never reads credentials and only permits one configured host on
# TCP 443. TLS remains end-to-end between the WSL client and the API endpoint.

param(
    [int]$ListenPort = 18765,
    [string]$AllowedHost = "llm.ki.k8s.rz.uni-potsdam.de"
)

$ErrorActionPreference = "Stop"

if ($ListenPort -lt 1024 -or $ListenPort -gt 65535) {
    throw "ListenPort must be between 1024 and 65535."
}
if (-not $AllowedHost) {
    throw "AllowedHost must not be empty."
}

Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading.Tasks;

public static class RestrictedConnectProxy
{
    private const int MaxHeaderBytes = 16384;

    public static void Run(int listenPort, string allowedHost)
    {
        var listener = new TcpListener(IPAddress.Loopback, listenPort);
        listener.Start();
        while (true)
        {
            TcpClient client = listener.AcceptTcpClient();
            Task.Run(() => Handle(client, allowedHost));
        }
    }

    private static void Handle(TcpClient client, string allowedHost)
    {
        using (client)
        {
            client.ReceiveTimeout = 10000;
            client.SendTimeout = 10000;
            NetworkStream downstream = client.GetStream();
            string header;
            try
            {
                header = ReadHeader(downstream);
            }
            catch
            {
                return;
            }

            string[] lines = header.Split(new[] { "\r\n" }, StringSplitOptions.None);
            string[] request = lines[0].Split(' ');
            if (request.Length < 2)
            {
                WriteResponse(downstream, "400 Bad Request", "");
                return;
            }

            if (request[0].Equals("GET", StringComparison.OrdinalIgnoreCase) &&
                request[1] == "/__ma_cc_health__")
            {
                WriteResponse(
                    downstream,
                    "204 No Content",
                    "Server: MA-CC-Potsdam-Bridge\r\n"
                );
                return;
            }

            if (!request[0].Equals("CONNECT", StringComparison.OrdinalIgnoreCase))
            {
                WriteResponse(downstream, "405 Method Not Allowed", "");
                return;
            }

            string authority = request[1];
            int separator = authority.LastIndexOf(':');
            int targetPort;
            if (separator <= 0 ||
                !Int32.TryParse(authority.Substring(separator + 1), out targetPort))
            {
                WriteResponse(downstream, "400 Bad Request", "");
                return;
            }
            string targetHost = authority.Substring(0, separator).Trim('[', ']');
            if (targetPort != 443 ||
                !targetHost.Equals(allowedHost, StringComparison.OrdinalIgnoreCase))
            {
                WriteResponse(downstream, "403 Forbidden", "");
                return;
            }

            using (var upstream = new TcpClient())
            {
                try
                {
                    Task connect = upstream.ConnectAsync(targetHost, targetPort);
                    if (!connect.Wait(15000))
                    {
                        WriteResponse(downstream, "504 Gateway Timeout", "");
                        return;
                    }
                    connect.GetAwaiter().GetResult();
                }
                catch
                {
                    WriteResponse(downstream, "502 Bad Gateway", "");
                    return;
                }

                WriteResponse(downstream, "200 Connection Established", "");
                using (NetworkStream upstreamStream = upstream.GetStream())
                {
                    Task toUpstream = downstream.CopyToAsync(upstreamStream);
                    Task toDownstream = upstreamStream.CopyToAsync(downstream);
                    Task.WaitAny(toUpstream, toDownstream);
                }
            }
        }
    }

    private static string ReadHeader(NetworkStream stream)
    {
        using (var buffer = new MemoryStream())
        {
            int matched = 0;
            byte[] terminator = { 13, 10, 13, 10 };
            while (buffer.Length < MaxHeaderBytes)
            {
                int value = stream.ReadByte();
                if (value < 0)
                    throw new EndOfStreamException();
                buffer.WriteByte((byte)value);
                if (value == terminator[matched])
                {
                    matched++;
                    if (matched == terminator.Length)
                        return Encoding.ASCII.GetString(buffer.ToArray());
                }
                else
                {
                    matched = value == terminator[0] ? 1 : 0;
                }
            }
            throw new InvalidDataException("HTTP proxy header is too large.");
        }
    }

    private static void WriteResponse(NetworkStream stream, string status, string headers)
    {
        byte[] response = Encoding.ASCII.GetBytes(
            "HTTP/1.1 " + status + "\r\n" + headers +
            "Content-Length: 0\r\nConnection: close\r\n\r\n"
        );
        stream.Write(response, 0, response.Length);
        stream.Flush();
    }
}
'@

[RestrictedConnectProxy]::Run($ListenPort, $AllowedHost)
