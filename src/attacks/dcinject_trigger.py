import torch
import torch.nn as nn
import torch.nn.functional as F

class DCINJECTTrigger:
    def __init__(self, device, attack_budget=0.03, noise_pattern="uniform",low_cutoff=0.1, high_cutoff=0.8, 
                 perceptual_weight=True, content_adaptive=True):
        self.device = device
        self.attack_budget = attack_budget
        self.noise_pattern = noise_pattern  
        self.mean_percent = 1.0
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.perceptual_weight = perceptual_weight
        self.content_adaptive = content_adaptive
        
    def formulate_trigger(self, image):       
        poisoned_image = image.clone()       
        nonzero_pixels = poisoned_image[poisoned_image > 0]
        
        if nonzero_pixels.numel() > 0:
            mean_value = torch.mean(nonzero_pixels.float()).to(poisoned_image.dtype)
            if poisoned_image.dtype == torch.uint8:
                poisoned_image = poisoned_image.to(torch.float32) - mean_value
            else:
                poisoned_image = poisoned_image - mean_value * self.mean_percent
            
            if self.noise_pattern == "uniform":
                noise = torch.empty_like(poisoned_image).uniform_(-self.attack_budget, self.attack_budget)
                poisoned_image = poisoned_image + noise
                
            elif self.noise_pattern == "gaussian":
                noise = torch.randn_like(poisoned_image) * self.attack_budget
                poisoned_image = poisoned_image + noise      
            
            if poisoned_image.dtype == torch.uint8:
                poisoned_image = torch.clamp(poisoned_image, min=0, max=255).to(torch.uint8)
            else:
                poisoned_image = torch.clamp(poisoned_image, min=0, max=1)
                
        return poisoned_image
    
    
    def frequency_domain_trigger(self, image):
        poisoned_image = image.clone()
        
        original_format_is_hwc = False
        
        if poisoned_image.dim() == 3 and poisoned_image.size(-1) == 3:
            poisoned_image = poisoned_image.permute(2, 0, 1)
            original_format_is_hwc = True
        
        orig_dtype = poisoned_image.dtype
        if poisoned_image.dtype == torch.uint8:
            poisoned_image = poisoned_image.to(torch.float32) / 255.0
        
        F_x_rgb = torch.fft.fft2(poisoned_image)        
        
        mean_freq = torch.mean(F_x_rgb, dim=[1, 2], keepdim=True)
        
        F_norm_rgb = F_x_rgb - mean_freq * self.mean_percent
        
        if self.noise_pattern == "uniform":
            noise_real = torch.empty_like(F_norm_rgb.real).uniform_(-self.attack_budget, self.attack_budget)
            #noise_imag = torch.empty_like(F_norm_rgb.imag).uniform_(-self.attack_budget, self.attack_budget)
        
        elif self.noise_pattern == "gaussian":
            noise_real = torch.randn_like(F_norm_rgb.real) * self.attack_budget
            # noise_imag = torch.randn_like(F_norm_rgb.imag) * self.attack_budget
        
        frequency_noise = torch.complex(noise_real, F_norm_rgb.imag)
        F_trigger_rgb = F_norm_rgb + frequency_noise
            
        # inverse FFT
        trigger_image = torch.fft.ifft2(F_trigger_rgb)
        
        if orig_dtype == torch.uint8:
            trigger_image = torch.clamp(trigger_image.real, min=0, max=255).to(torch.uint8)
        else:
            trigger_image = torch.clamp(trigger_image.real, min=0, max=1)
        
        # Restore original format
        if original_format_is_hwc:
            trigger_image = trigger_image.permute(1, 2, 0)
            
        return trigger_image
    
    
    def apply_trigger_batch_frequency(self, images, labels, target_label, poison_ratio):
            batch_size = images.size(0)
            poison_mask = torch.rand(batch_size, device=images.device) <= poison_ratio
            
            if poison_mask.sum().item() == 0:
                return images, labels

            poisoned_images = images.clone()
            poisoned_labels = torch.full([batch_size], target_label, device=labels.device)

            for i in range(batch_size):
                if poison_mask[i]:
                    poisoned_images[i] = self.frequency_domain_trigger(poisoned_images[i])

            final_images = poison_mask.view(-1, 1, 1, 1).float() * poisoned_images + \
                        (~poison_mask.view(-1, 1, 1, 1)).float() * images
            final_labels = poison_mask.float() * poisoned_labels + (~poison_mask).float() * labels

            return final_images, final_labels.to(torch.long)    
        
            
            
    def apply_trigger_batch(self, images, labels, target_label, poison_ratio):        
        batch_size = images.size(0)
        poison_mask = torch.rand(batch_size, device=images.device) <= poison_ratio
        
        if poison_mask.sum().item() == 0:
            return images, labels

        poisoned_images = images.clone()
        poisoned_labels = torch.full([batch_size], target_label, device=labels.device)

        for i in range(batch_size):
            if poison_mask[i]:
                poisoned_images[i] = self.formulate_trigger(poisoned_images[i])

        final_images = poison_mask.view(-1, 1, 1, 1).float() * poisoned_images + \
                      (~poison_mask.view(-1, 1, 1, 1)).float() * images
        final_labels = poison_mask.float() * poisoned_labels + (~poison_mask).float() * labels

        return final_images, final_labels.to(torch.long) 
    
    def create_frequency_mask(self,shape,device):
        H, W = shape[-2:]
        
        u = torch.fft.fftfreq(H, device=device).unsqueeze(1)
        v = torch.fft.fftfreq(W, device=device).unsqueeze(0)
        
        # distance from DC component
        freq_dist = torch.sqrt(u**2 + v**2)
        
        mask = (freq_dist > self.low_cutoff) & (freq_dist < self.high_cutoff)
        
        return mask.float()
    
    
    def perceptual_frequency_weight(self,F_x):
        H, W = F_x.shape[-2:]
        device = F_x.device
        
        # frequency grid
        u = torch.fft.fftfreq(H, device=device).unsqueeze(0)
        v = torch.fft.fftfreq(W, device=device).unsqueeze(1)
        
        freq_dist = torch.sqrt(u**2 + v**2)
        
        hvs_senstivity = torch.exp(-freq_dist*3.0)
        perceptual_weight = 1.0 / (0.01 + hvs_senstivity)
        perceptual_weight = perceptual_weight / torch.sum(perceptual_weight)
        
        return perceptual_weight
    
    def adaptive_frequency_selection(self,F_x):
        H, W = F_x.shape[-2:]
        center_h, center_w = H//2, W//2
        
        high_freq_region = F_x.clone()
        
        mask_size  = min(H,W)//4 
        
        high_freq_region[:,
                         center_h-mask_size:center_h + mask_size,
                         center_w-mask_size:center_w+mask_size] = 0  
        
        # texture indicator
        high_freq_energy = torch.sum(torch.abs(high_freq_region)**2,dim=[-2,-1])
        
        total_energy = torch.sum(torch.abs(F_x)**2, dim=[-2,-1])   
        
        texture_ratio = high_freq_energy / (total_energy + 1e-8)
        
        adaptive_factor = 0.5+1.5*texture_ratio
        
        return adaptive_factor.unsqueeze(-1).unsqueeze(-1)
    
    
    def frequency_domain_adaptive_trigger(self, image):
        """
        Advanced frequency domain trigger incorporating signal processing principles
        """
        poisoned_image = image.clone()
        
        # Handle input format
        original_format_is_hwc = False
        if poisoned_image.dim() == 3 and poisoned_image.size(-1) == 3:
            poisoned_image = poisoned_image.permute(2, 0, 1)
            original_format_is_hwc = True
        
        # Handle data type
        orig_dtype = poisoned_image.dtype
        if poisoned_image.dtype == torch.uint8:
            poisoned_image = poisoned_image.to(torch.float32) / 255.0
        
        # 2D FFT transform
        F_x_rgb = torch.fft.fft2(poisoned_image)
        
        # 1. DC component handling (your original approach)
        mean_freq = torch.mean(F_x_rgb, dim=[1, 2], keepdim=True)
        F_norm_rgb = F_x_rgb - mean_freq * self.mean_percent
        
        # 2. Create frequency-selective mask (mid-frequencies)
        freq_mask = self.create_frequency_mask(F_norm_rgb.shape, F_norm_rgb.device)
        
        # 3. Perceptual weighting based on HVS
        if self.perceptual_weight:
            perceptual_weights = self.perceptual_frequency_weight(F_norm_rgb)
        else:
            perceptual_weights = torch.ones_like(freq_mask)
        
        # 4. Content-adaptive scaling
        if self.content_adaptive:
            adaptive_factors = self.adaptive_frequency_selection(F_norm_rgb)
        else:
            adaptive_factors = torch.ones(F_norm_rgb.shape[0], 1, 1, 
                                        device=F_norm_rgb.device)
        
        # 5. Generate adaptive noise
        # Base Gaussian noise
        noise_real = torch.randn_like(F_norm_rgb.real) * self.attack_budget
        noise_imag = torch.randn_like(F_norm_rgb.imag) * self.attack_budget
        
        # Apply all adaptive weightings
        # freq_mask: Target mid-frequencies only
        # perceptual_weights: Embed more where HVS is less sensitive  
        # adaptive_factors: Scale based on image texture content
        combined_weight = freq_mask * perceptual_weights * adaptive_factors
        
        noise_real = noise_real * combined_weight
        noise_imag = noise_imag * combined_weight
        
        frequency_noise = torch.complex(noise_real, noise_imag)
        
        # 6. Apply adaptive trigger
        F_trigger_rgb = F_norm_rgb + frequency_noise
        
        # 7. Inverse FFT
        trigger_image = torch.fft.ifft2(F_trigger_rgb)
        
        # Handle output format
        if orig_dtype == torch.uint8:
            trigger_image = torch.clamp(trigger_image.real * 255, min=0, max=255).to(torch.uint8)
        else:
            trigger_image = torch.clamp(trigger_image.real, min=0, max=1)
        
        # Restore original format
        if original_format_is_hwc:
            trigger_image = trigger_image.permute(1, 2, 0)
            
        return trigger_image
    
    def get_embedding_statistics(self, image):
        """
        Analyze where and how much the trigger is being embedded
        Useful for understanding the adaptive behavior
        """
        # Convert to working format
        if image.dim() == 3 and image.size(-1) == 3:
            image = image.permute(2, 0, 1)
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0
        
        F_x = torch.fft.fft2(image)
        
        # Get all adaptive components
        freq_mask = self.create_frequency_mask(F_x.shape, F_x.device)
        perceptual_weights = self.perceptual_frequency_weight(F_x)
        adaptive_factors = self.adaptive_frequency_selection(F_x)
        
        combined_weight = freq_mask * perceptual_weights * adaptive_factors
        
        stats = {
            'freq_mask_active_ratio': torch.mean(freq_mask).item(),
            'avg_perceptual_weight': torch.mean(perceptual_weights).item(),
            'adaptive_factor': torch.mean(adaptive_factors).item(),
            'combined_weight_mean': torch.mean(combined_weight).item(),
            'combined_weight_std': torch.std(combined_weight).item(),
            'embedding_regions': torch.sum(combined_weight > 0.1).item() / combined_weight.numel()
        }
        
        return stats
    
    def apply_trigger_batch_adaptive_frequency(self, images, labels, target_label, poison_ratio):
        batch_size = images.size(0)
        poison_mask = torch.rand(batch_size, device=images.device) <= poison_ratio
        
        if poison_mask.sum().item() == 0:
            return images, labels
            
        poisoned_images = images.clone()
        poisoned_labels = torch.full([batch_size], target_label, device=labels.device)
        
        for i in range(batch_size):
            if poison_mask[i]:
                poisoned_images[i] = self.frequency_domain_adaptive_trigger(poisoned_images[i])
                
        final_images = poison_mask.view(-1, 1, 1, 1).float() * poisoned_images + \
                      (~poison_mask.view(-1, 1, 1, 1)).float() * images
        final_labels = poison_mask.float() * poisoned_labels + (~poison_mask).float() * labels

        return final_images, final_labels.to(torch.long)
        
        
        
        
        
      
        
        
      
        
       
    
