import { ArrowRight, Camera, UploadCloud, AudioLines, ShieldCheck, Zap, Sun, Moon } from 'lucide-react'
import useSignStore from '../../store/useSignStore'
import logoDark from '../../assets/logo_dark.png'
import logoLight from '../../assets/logo_light.png'
import heroImg from '../../assets/hero.jpg'

export default function LandingPage({ onStart }) {
  const theme = useSignStore(s => s.theme)
  const toggleTheme = useSignStore(s => s.toggleTheme)

  const logoSrc = theme === 'dark' ? logoLight : logoDark

  return (
    <div className="min-h-screen w-full bg-slate-50 text-slate-800 dark:bg-slate-950 dark:text-slate-100 font-sans transition-colors duration-300 relative overflow-x-hidden">
      
      <div className="absolute inset-0 bg-[linear-gradient(to_right,#0f172a04_1px,transparent_1px),linear-gradient(to_bottom,#0f172a04_1px,transparent_1px)] dark:bg-[linear-gradient(to_right,#ffffff02_1px,transparent_1px),linear-gradient(to_bottom,#ffffff02_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)] pointer-events-none" />

      <div className="absolute top-[-10%] left-[-10%] w-[600px] h-[600px] bg-indigo-500/10 dark:bg-indigo-500/15 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[10%] right-[-10%] w-[600px] h-[600px] bg-purple-500/10 dark:bg-purple-500/15 rounded-full blur-[120px] pointer-events-none" />

      {/* AI Keypoint Node indicators representing Sign Language camera tracking */}
      <div className="absolute top-[15%] left-[8%] w-1.5 h-1.5 rounded-full bg-indigo-500/40 dark:bg-indigo-400/50 animate-pulse pointer-events-none" />
      <div className="absolute top-[40%] right-[15%] w-2 h-2 rounded-full bg-purple-500/30 dark:bg-purple-400/40 animate-ping pointer-events-none" style={{ animationDuration: '3s' }} />
      <div className="absolute bottom-[20%] left-[25%] w-1.5 h-1.5 rounded-full bg-emerald-500/30 dark:bg-emerald-400/40 pointer-events-none" />
      <div className="absolute top-[60%] left-[12%] w-2 h-2 rounded-full bg-pink-500/30 dark:bg-pink-400/40 pointer-events-none" />

      <header className="sticky top-0 z-50 w-full border-b border-slate-200/60 dark:border-slate-900/60 bg-white/60 dark:bg-slate-950/60 backdrop-blur-md transition-colors duration-300">
        <div className="w-full px-8 lg:px-[5%] h-20 flex items-center justify-between">
          <div className="flex items-center gap-3 select-none">
            <img src={logoSrc} alt="Logo" className="h-10 md:h-12 w-auto object-contain rounded-xl transition-all duration-300 hover:scale-105" />
            <div className="flex flex-col">
              <span className="text-lg font-bold tracking-tight text-slate-800 dark:text-white bg-gradient-to-r from-indigo-500 to-purple-500 bg-clip-text text-transparent">
                SignSpeak AI
              </span>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 font-medium tracking-wide mt-0.5">
                Hệ thống nhận dạng Ký hiệu Tiếng Việt
              </span>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={toggleTheme}
              className="p-2.5 rounded-xl text-slate-500 hover:text-indigo-500 dark:text-slate-400 dark:hover:text-indigo-400 bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 transition-all duration-200 shadow-sm"
              title={theme === 'dark' ? 'Chuyển sang giao diện sáng' : 'Chuyển sang giao diện tối'}
            >
              {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
          </div>
        </div>
      </header>

      <section className="w-full px-8 lg:px-[5%] py-12 lg:pt-6 lg:pb-16 lg:h-[calc(100vh-5rem)] flex flex-col gap-10 lg:gap-6 relative z-10">

        <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[minmax(0,42fr)_minmax(0,58fr)] gap-12 lg:gap-10 items-center">

          <div className="flex flex-col gap-6 lg:gap-12 2xl:gap-16 text-left">
          <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs md:text-sm font-extrabold border border-emerald-500/20 w-fit">
            <Zap size={12} />
            <span>AI Đột Phá - Biên Dịch Cử Chỉ</span>
          </div>

          {/* Font size uses clamp() to fit the left column; gap (not line-height) keeps 3-line spacing precise */}
          <h1 className="flex flex-col gap-4 xl:gap-6 font-extrabold tracking-tight leading-[1.05] pb-1 text-slate-800 dark:text-white text-4xl md:text-5xl lg:text-[clamp(2rem,3.3vw,4rem)] lg:whitespace-nowrap">
            <span>Thu hẹp</span>
            <span>khoảng cách giao tiếp</span>
            {/* w-fit keeps the gradient tight to the text width */}
            <span className="w-fit text-transparent bg-clip-text bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500">
              bằng Trí tuệ nhân tạo
            </span>
          </h1>

          <p className="text-slate-600 dark:text-slate-300 text-base md:text-lg lg:text-xl leading-relaxed max-w-[620px]">
            SignSpeak AI là giải pháp biên dịch ngôn ngữ ký hiệu tiếng Việt sang văn bản và giọng nói thời gian thực.
            Hỗ trợ nhận diện qua camera trực tiếp hoặc tải lên video lưu trữ một cách chính xác và thuận tiện.
          </p>

          <div className="flex flex-col sm:flex-row gap-4">
            <button
              onClick={onStart}
              className="flex items-center justify-center gap-2 px-8 py-4 rounded-xl bg-indigo-500 hover:bg-indigo-600 text-base md:text-lg font-bold text-white shadow-xl shadow-indigo-500/25 hover:scale-[1.03] active:scale-95 transition-all duration-200"
            >
              <span>Bắt đầu phiên dịch</span>
              <ArrowRight size={18} />
            </button>
            <button
              type="button"
              onClick={onStart}
              className="flex items-center justify-center gap-2 px-8 py-4 rounded-xl bg-slate-200/50 hover:bg-slate-200 dark:bg-slate-900 dark:hover:bg-slate-800 border border-slate-300 dark:border-slate-800 text-base md:text-lg font-bold text-slate-700 dark:text-slate-300 transition-all duration-200"
            >
              Tìm hiểu thêm
            </button>
          </div>
        </div>

        <div className="relative flex items-center justify-center w-full">
          <div className="absolute inset-0 bg-gradient-to-tr from-indigo-500/20 to-purple-500/20 rounded-[50px] blur-3xl opacity-50 scale-110 animate-[pulse_6s_infinite]" />
          <div className="relative w-full border border-slate-200/80 dark:border-slate-900/80 p-3 rounded-[28px] bg-white/30 dark:bg-slate-900/25 backdrop-blur-md shadow-2xl transition-all hover:scale-[1.01] duration-300">
            <img
              src={heroImg}
              alt="SignSpeak AI Hero Illustration"
              className="w-full aspect-[16/10] rounded-[20px] object-cover shadow-lg border border-slate-100/10"
            />
          </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-x-8 gap-y-6 lg:flex lg:justify-between lg:gap-x-6 2xl:w-4/5 2xl:mx-auto">
          <div className="flex items-center gap-3">
            <div className="shrink-0 p-3 rounded-xl bg-slate-200/50 dark:bg-slate-900/50 border border-slate-300/60 dark:border-slate-800 text-indigo-500 dark:text-indigo-400">
              <Camera size={20} />
            </div>
            <div className="flex flex-col">
              <span className="text-sm xl:text-base font-bold text-slate-800 dark:text-white leading-snug lg:whitespace-nowrap">Nhận diện thời gian thực</span>
              <span className="text-xs md:text-sm text-slate-500 dark:text-slate-400">Camera trực tiếp</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="shrink-0 p-3 rounded-xl bg-slate-200/50 dark:bg-slate-900/50 border border-slate-300/60 dark:border-slate-800 text-indigo-500 dark:text-indigo-400">
              <UploadCloud size={20} />
            </div>
            <div className="flex flex-col">
              <span className="text-sm xl:text-base font-bold text-slate-800 dark:text-white leading-snug lg:whitespace-nowrap">Tải lên video</span>
              <span className="text-xs md:text-sm text-slate-500 dark:text-slate-400">Hỗ trợ nhiều định dạng</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="shrink-0 p-3 rounded-xl bg-slate-200/50 dark:bg-slate-900/50 border border-slate-300/60 dark:border-slate-800 text-indigo-500 dark:text-indigo-400">
              <AudioLines size={20} />
            </div>
            <div className="flex flex-col">
              <span className="text-sm xl:text-base font-bold text-slate-800 dark:text-white leading-snug lg:whitespace-nowrap">Chuyển văn bản & giọng nói</span>
              <span className="text-xs md:text-sm text-slate-500 dark:text-slate-400">Đa dạng giọng đọc</span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="shrink-0 p-3 rounded-xl bg-slate-200/50 dark:bg-slate-900/50 border border-slate-300/60 dark:border-slate-800 text-indigo-500 dark:text-indigo-400">
              <ShieldCheck size={20} />
            </div>
            <div className="flex flex-col">
              <span className="text-sm xl:text-base font-bold text-slate-800 dark:text-white leading-snug lg:whitespace-nowrap">Chính xác & bảo mật</span>
              <span className="text-xs md:text-sm text-slate-500 dark:text-slate-400">Độ tin cậy cao</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
