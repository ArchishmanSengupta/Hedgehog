# Documentation: https://docs.brew.sh/Formula-Cookbook
#                https://rubydoc.brew.sh/Formula
# PLEASE REMOVE ALL GENERATED COMMENTS BEFORE SUBMITTING YOUR PULL REQUEST!

class HedgehogDlm < Formula
  desc "Lightweight framework for training and fine-tuning Diffusion Language Models"
  homepage "https://github.com/ArchishmanSengupta/Hedgehog"
  url "https://files.pythonhosted.org/packages/source/h/hedgehog-dlm/hedgehog_dlm-0.2.0.tar.gz"
  sha256 "1d3e5d1d2d5655d83889bdd7cccde42107500e95c7d45e79998810a32f57887c"
  license "Apache-2.0"
  version "0.2.0"

  depends_on "python@3.10"

  def install
    python = "python3.10"
    virtualenv_create(libexec, python)
    virtualenv_pip_libdir = libexec/"lib/python3.10/site-packages"
    ENV["PYTHONPATH"] = virtualenv_pip_libdir

    system "#{libexec}/bin/pip", "install", "hedgehog-dlm==#{version}"

    bin.install_symlink Dir["#{libexec}/bin/*"].select { |f| f.executable? }
  end

  test do
    system "#{bin}/hedgehog", "--help"
  end
end
