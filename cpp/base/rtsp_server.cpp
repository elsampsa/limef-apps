/*
 * rtsp_server - Interactive RTSP server for testing
 *
 * Usage:
 *   ./rtsp_server --file /path/to/video.mkv --port 8554
 *
 * Then connect with:
 *   ffplay rtsp://localhost:8554/live/stream
 *   ffplay -rtsp_transport tcp rtsp://localhost:8554/live/stream
 *   vlc rtsp://localhost:8554/live/stream
 *
 * Press Ctrl+C to stop.
 */

#include <iostream>
#include <string>
#include <atomic>
#include <csignal>
#include <thread>
#include <chrono>
#include <getopt.h>

#include "limef/rtsp/rtspserverthread.h"
#include "limef/framefilter/rtp.h"
#include "limef/thread/mediafile.h"

// Global flag for Ctrl+C handling
static std::atomic<bool> g_running{true};

static void signal_handler(int signum) {
    if (signum == SIGINT || signum == SIGTERM) {
        std::cout << "\nShutting down..." << std::endl;
        g_running = false;
    }
}

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " --file <media_file> [--port <port>]\n"
              << "\n"
              << "Options:\n"
              << "  --file, -f <path>   Media file to stream (required)\n"
              << "  --port, -p <port>   RTSP port (default: 8554)\n"
              << "  --help, -h          Show this help\n"
              << "\n"
              << "Example:\n"
              << "  " << prog << " --file video.mkv --port 8554\n"
              << "\n"
              << "Then connect with:\n"
              << "  ffplay rtsp://localhost:8554/live/stream\n"
              << "  ffplay -rtsp_transport tcp rtsp://localhost:8554/live/stream\n"
              << "  vlc rtsp://localhost:8554/live/stream\n"
              << std::endl;
}

int main(int argc, char** argv) {
    // ffmpeg_av_log_set_level(100);

    std::string media_file;
    int port = 8554;

    // Parse command line arguments
    static struct option long_options[] = {
        {"file", required_argument, 0, 'f'},
        {"port", required_argument, 0, 'p'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "f:p:h", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'f':
                media_file = optarg;
                break;
            case 'p':
                port = std::stoi(optarg);
                break;
            case 'h':
            default:
                print_usage(argv[0]);
                return (opt == 'h') ? 0 : 1;
        }
    }

    if (media_file.empty()) {
        std::cerr << "Error: --file is required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Set up signal handler
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    using namespace Limef;
    using namespace Limef::thread;
    using namespace Limef::ff;

    const int SLOT = 1;
    const char* URL_TAIL = "/live/stream";

    std::cout << "=================================\n";
    std::cout << "  RTSP Server\n";
    std::cout << "=================================\n";
    std::cout << "Media file: " << media_file << "\n";
    std::cout << "Port:       " << port << "\n";
    std::cout << "URL:        rtsp://localhost:" << port << URL_TAIL << "\n";
    std::cout << "=================================\n";
    std::cout << "Press Ctrl+C to stop\n\n";

    // Create media file context - looping at native fps
    MediaFileContext media_ctx(media_file.c_str(), SLOT);
    media_ctx.fps = -1;   // Native playback speed
    // media_ctx.loop = 100; // Loop with 100ms pause at EOF
    media_ctx.loop = 0;

    FrameFifoContext buffer_ctx(false, 32, 64);

    // Create threads and filters
    MediaFileThread mediafile("mediafile", media_ctx, buffer_ctx);
    // mediafile.setLogLevel(spdlog::level::trace);
    RTPMuxerFrameFilter rtp_muxer("rtp_muxer");
    Limef::rtsp::RTSPServerThread rtsp_server(
        "rtsp_server",
        FrameFifoContext(false, 5, 100),
        port
    );

    // Connect filterchain: mediafile -> rtp_muxer -> rtsp_server
    mediafile.getOutput().cc(rtp_muxer).cc(rtsp_server.getInput());

    // Connect callbacks for logging
    rtsp_server.streamRequired.connect([](int slot) {
        std::cout << "[event] Client subscribed to slot " << slot << std::endl;
    });
    rtsp_server.streamNotRequired.connect([](int slot) {
        std::cout << "[event] No more clients on slot " << slot << std::endl;
    });

    // Start RTSP server
    std::cout << "Starting RTSP server..." << std::endl;
    rtsp_server.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Expose the stream
    rtsp_server.expose(SLOT, URL_TAIL);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Start media playback
    std::cout << "Starting media playback..." << std::endl;
    mediafile.start();

    // Wait for initialization
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "\nReady! Connect with:\n";
    std::cout << "  ffplay rtsp://localhost:" << port << URL_TAIL << "\n";
    std::cout << "  ffplay -rtsp_transport tcp rtsp://localhost:" << port << URL_TAIL << "\n\n";

    // Main loop - wait for Ctrl+C
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // Cleanup
    std::cout << "Stopping media playback..." << std::endl;
    mediafile.stop();

    std::cout << "Stopping RTSP server..." << std::endl;
    rtsp_server.stop();

    std::cout << "Done." << std::endl;
    return 0;
}
